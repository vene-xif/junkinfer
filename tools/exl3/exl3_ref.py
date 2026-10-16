#!/usr/bin/env python3
"""Reference EXL3 decoder (numpy), mirroring exllamav3's reconstruct_tile (exl3_dq.cuh / codebook.cuh / reconstruct.cu).

A linear stores: trellis int16 [K/16, N/16, 16*bits] (one 16x16 tile = 256 weights = 16*bits uint16),
suh fp16 [K], svh fp16 [N], mcg multiplier (codebook "mcg"). Inner weight W_hat[K][N] (row = input index).
Original weight (K x N): W = diag(suh) . H . W_hat . H . diag(svh), H = blockwise Sylvester Hadamard 128 / sqrt(128),
so y = x W = H128(H128(x * suh) W_hat) * svh.

usage: exl3_ref.py MODEL_DIR TENSOR_PREFIX [GGUF GGUF_TENSOR]   (cosine similarity vs the GGUF's dequantised weight)
"""
import json, os, struct, sys
import numpy as np

def st_index(model_dir):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    return idx

def st_read(model_dir, idx, name):
    fn = os.path.join(model_dir, idx[name])
    with open(fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        meta = hdr[name]
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        raw = f.read(b - a)
    dt = {"I16": np.int16, "F16": np.float16, "I32": np.int32, "BF16": np.uint16, "F32": np.float32}[meta["dtype"]]
    arr = np.frombuffer(raw, dtype=dt).reshape(meta["shape"])
    if meta["dtype"] == "BF16":
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr

def mcg_decode(w16, mult):
    x = (w16.astype(np.uint64) * np.uint64(mult)) & np.uint64(0xFFFFFFFF)
    x = x.astype(np.uint32)
    x = (x & np.uint32(0x8fff8fff)) ^ np.uint32(0x3b603b60)
    lo = (x & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (x >> np.uint32(16)).astype(np.uint16).view(np.float16).astype(np.float32)
    return (lo + hi).astype(np.float16).astype(np.float32)   # the kernels add in fp16

def tile_pos_map():
    """position p (0..255) in the trellis -> (row, col) in the 16x16 tile (tensor-core fragment shuffle)"""
    rc = np.zeros((256, 2), dtype=np.int64)
    for L in range(32):
        base, odd = L & ~4, (L >> 2) & 1
        r0, c0 = (base % 4) * 2, base // 8
        rows = [r0, r0 + 1, r0 + 8, r0 + 9]
        for i in range(8):
            r = rows[i % 4]
            cp = c0 if i < 4 else c0 + 4
            rc[L * 8 + i] = (r, 2 * cp + odd)
    assert len({(int(a), int(b)) for a, b in rc}) == 256
    return rc

def decode_inner(trellis, mult):
    kt, nt, words16 = trellis.shape
    bits = words16 // 16
    L = 256 * bits
    u16 = trellis.astype(np.uint16).reshape(kt, nt, words16)
    w32 = (u16[..., 0::2].astype(np.uint32) | (u16[..., 1::2].astype(np.uint32) << 16))     # [kt, nt, 8*bits]
    # MSB-first bit array per tile
    be = w32.astype(">u4").view(np.uint8).reshape(kt, nt, -1)
    bitarr = np.unpackbits(be, axis=-1)                                                       # [kt, nt, L]
    p = np.arange(256)
    start = ((p + 257) * bits - 16) % L
    idx = (start[:, None] + np.arange(16)[None, :]) % L                                      # [256, 16]
    weights = (1 << (15 - np.arange(16))).astype(np.uint32)
    win = (bitarr[..., idx].astype(np.uint32) * weights).sum(-1).astype(np.uint32)           # [kt, nt, 256]
    vals = mcg_decode(win, mult)
    rc = tile_pos_map()
    W = np.zeros((kt, 16, nt, 16), dtype=np.float32)
    W[:, rc[:, 0], :, rc[:, 1]] = np.moveaxis(vals, 2, 0)                                    # [256, kt, nt] -> place
    return W.reshape(kt * 16, nt * 16)

def sylvester(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h

def full_weight(model_dir, prefix):
    idx = st_index(model_dir)
    tr = st_read(model_dir, idx, prefix + ".trellis")
    suh = st_read(model_dir, idx, prefix + ".suh").astype(np.float32)
    svh = st_read(model_dir, idx, prefix + ".svh").astype(np.float32)
    mcg = int(st_read(model_dir, idx, prefix + ".mcg").astype(np.uint32).reshape(-1)[0])
    Wi = decode_inner(tr, mcg)
    H = sylvester(128) / np.sqrt(128)
    K, N = Wi.shape
    W = (H @ Wi.reshape(K // 128, 128, N)).reshape(K, N)
    W = W * suh[:, None]
    W = (W.reshape(K, N // 128, 128) @ H).reshape(K, N)
    W = W * svh[None, :]
    return W, mcg

if __name__ == "__main__":
    model_dir, prefix = sys.argv[1], sys.argv[2]
    W, mcg = full_weight(model_dir, prefix)
    print(f"{prefix}: W {W.shape}, mcg {mcg:#x}, rms {np.sqrt((W**2).mean()):.5f}")
    if len(sys.argv) > 4:
        sys.path.insert(0, os.path.expanduser("~/junkinfer/engine/gguf-py"))
        import gguf
        r = gguf.GGUFReader(sys.argv[3])
        t = [t for t in r.tensors if t.name == sys.argv[4]][0]
        G = gguf.quants.dequantize(t.data, t.tensor_type).reshape(-1, t.shape[0]).astype(np.float32)   # [out, in]
        G = G.T
        print(f"gguf {sys.argv[4]}: {G.shape}, rms {np.sqrt((G**2).mean()):.5f}")
        cos = (W * G).sum() / np.sqrt((W * W).sum() * (G * G).sum())
        print(f"cosine similarity {cos:.5f}; rel. diff rms {np.sqrt(((W - G)**2).mean()) / np.sqrt((G**2).mean()):.4f}")

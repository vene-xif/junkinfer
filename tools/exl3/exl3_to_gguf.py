#!/usr/bin/env python3
"""Convert an exllamav3 EXL3 Qwen3.5/3.6-MoE (qwen35moe) or dense Qwen3.5/3.8 (qwen35) checkpoint to a JunkInfer GGUF.

EXL3 linears become GGML_TYPE_EXL3_<bits> tensors (one ggml block = one 16x16 trellis tile, tiles re-ordered from
exllamav3's [K/16][N/16] to [N/16][K/16] so a group of 16 outputs is contiguous) plus F32 `<name>.suh` [K] and
`<name>.svh` [N] (experts: [E, K] / [E, N]). The forward pass of such a weight is y = H128(H128(x*suh) W_hat)*svh.
Other tensors get llama.cpp's qwen35 conversion transforms (A_log -> -exp, RMSNorm +1, conv1d squeeze) and the
GDN value heads are reordered grouped -> tiled like convert_hf_to_gguf.py does; on EXL3 tensors whole 128-wide
heads are moved (tile groups + scale vectors), which commutes with the blockwise Hadamard.
Metadata (hparams, tokenizer, template) is copied from an existing qwen35moe / qwen35 GGUF of the same architecture; the MTP
layer is dropped. Fine-tunes with their own tokenizer.json (same BPE, e.g. MiniCPM-V 4.7: Qwen3.6's vocabulary plus
special tokens in its padding slots) get the token list / types re-derived from it, truncated to the embedding rows,
and their chat template. Codebook: mcg or mul1 (per tensor; must be uniform) -> junkinfer.exl3.codebook 0 / 1.
Any width 2..8 bits per weight (GGML_TYPE_EXL3_<bits>). n_vocab = the head's rows (a multiple of 128: its output
Hadamard blocks need every row); a shorter embedding gets zero rows and the extra ids stay [PAD] tokens.

usage: exl3_to_gguf.py EXL3_DIR META_GGUF OUT_GGUF
"""
import json, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/junkinfer/engine/gguf-py"))
import gguf  # noqa: E402

MCG = 0xCBAC1FED
HB = 128   # Hadamard block
CODEBOOK = {"mcg": 0, "mul1": 1}


class ST:
    def __init__(self, d):
        self.d = d
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}
        for fn in sorted(set(self.idx.values())):
            with open(os.path.join(d, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            mm = np.memmap(os.path.join(d, fn), dtype=np.uint8, mode="r")
            for k, v in h.items():
                if k != "__metadata__":
                    self.hdr[k] = (mm, 8 + n, v)

    def has(self, name):
        return name in self.hdr

    def get(self, name):
        mm, base, meta = self.hdr[name]
        a, b = meta["data_offsets"]
        dt = {"I16": np.int16, "F16": np.float16, "I32": np.int32, "BF16": np.uint16, "F32": np.float32}[meta["dtype"]]
        arr = mm[base + a: base + b].view(dt).reshape(meta["shape"])
        if meta["dtype"] == "BF16":
            return arr, "bf16"
        return arr, None

    def f32(self, name):
        arr, kind = self.get(name)
        if kind == "bf16":
            return (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32)


def block_perm(nk, nvpk):
    """grouped -> tiled order of value heads: new head j = old head p[j]"""
    return np.arange(nk * nvpk).reshape(nk, nvpk).T.reshape(-1)


def main():
    src, meta_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    st = ST(src)
    cfg = json.load(open(os.path.join(src, "config.json")))
    tc = cfg.get("text_config", cfg)
    n_layer = tc["num_hidden_layers"]
    n_expert = tc.get("num_experts")          # None: dense (qwen35, e.g. Qwen3.8 27B)
    nk, nv = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    hk, hv = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    assert hv == HB, "head reorder of EXL3 tensors needs 128-wide value heads"
    vperm = block_perm(nk, nv // nk)          # in units of value heads (= Hadamard blocks)
    full_int = tc.get("full_attention_interval", 4)
    P = "model.language_model."
    P_EMB = P + "embed_tokens.weight"
    cbs = {k.rsplit(".", 1)[1] for k in st.hdr if k.endswith((".mcg", ".mul1"))}
    assert len(cbs) == 1, f"mixed codebooks {cbs}"
    cb = cbs.pop()

    rd = gguf.GGUFReader(meta_path)
    arch = rd.fields["general.architecture"].contents()
    assert arch == ("qwen35moe" if n_expert else "qwen35"), arch
    w = gguf.GGUFWriter(out_path, arch)
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture",
            f"{arch}.nextn_predict_layers", f"{arch}.block_count", "general.file_type"}
    n_emb = st.hdr[P_EMB][2]["shape"][0]
    n_vocab = max(n_emb, st.hdr["lm_head.trellis"][2]["shape"][1] * 16)   # lm_head trellis [K/16, N/16, 16 bits]
    assert len(rd.fields["tokenizer.ggml.tokens"].data) >= n_vocab, "meta GGUF has fewer tokens than the head"
    over = {}   # tokenizer fields re-derived from the checkpoint's own tokenizer.json
    tj_path = os.path.join(src, "tokenizer.json")
    if os.path.exists(tj_path):
        tj = json.load(open(tj_path))
        ids = dict(tj["model"]["vocab"])
        special = {}
        for a in tj.get("added_tokens", []):
            ids[a["content"]] = a["id"]
            special[a["id"]] = a.get("special", False)
        inv = {v: k for k, v in ids.items()}
        toks = rd.fields["tokenizer.ggml.tokens"].contents()[:n_vocab]
        types = rd.fields["tokenizer.ggml.token_type"].contents()[:n_vocab]
        changed = 0
        for i in range(n_vocab):
            t = inv.get(i)
            if t is not None and t != toks[i]:
                toks[i] = t
                types[i] = 3 if special.get(i, False) else 4   # CONTROL / USER_DEFINED
                changed += 1
        over["tokenizer.ggml.tokens"] = toks
        over["tokenizer.ggml.token_type"] = types
        print(f"tokenizer: {n_vocab} tokens, {changed} re-derived from {tj_path}")
    ct = os.path.join(src, "chat_template.jinja")
    if os.path.exists(ct):
        over["tokenizer.chat_template"] = open(ct).read()
    for f in rd.fields.values():
        if f.name in skip or f.name.startswith("quantize."):
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, over.get(f.name, f.contents()), vt, sub_type=sub)
    w.add_key_value(f"{arch}.block_count", n_layer, gguf.GGUFValueType.UINT32)
    w.add_key_value("junkinfer.exl3.version", 1, gguf.GGUFValueType.UINT32)
    w.add_key_value("junkinfer.exl3.mcg", MCG, gguf.GGUFValueType.UINT32)
    w.add_key_value("junkinfer.exl3.codebook", CODEBOOK[cb], gguf.GGUFValueType.UINT32)
    w.add_key_value("junkinfer.exl3.source", os.path.basename(os.path.normpath(src)), gguf.GGUFValueType.STRING)

    plan = []   # (name, np_shape, dtype for info, nbytes, raw_dtype, producer)

    def add_f32(name, fn, shape):
        n = int(np.prod(shape)) * 4
        plan.append((name, list(shape), np.float32, n, None, lambda: np.ascontiguousarray(fn(), dtype=np.float32).reshape(shape)))

    def add_f16(name, fn, shape):
        n = int(np.prod(shape)) * 2
        def get():
            a = np.ascontiguousarray(fn())
            assert a.dtype == np.float16, (name, a.dtype)
            return a.reshape(shape)
        plan.append((name, list(shape), np.float16, n, None, get))

    def exl3_parts(prefix):
        tr, _ = st.get(prefix + ".trellis")
        if cb == "mcg":
            mcg = int(np.asarray(st.get(prefix + ".mcg")[0]).astype(np.int64).reshape(-1)[0]) & 0xFFFFFFFF
            assert mcg == MCG, f"{prefix}: mcg {mcg:#x}"
        else:
            assert st.has(prefix + ".mul1"), prefix
        return tr

    def exl3_one(prefix, nperm=None, kperm=None):
        """-> (tiles [N/16, K/16, 16*bits] int16, suh [K], svh [N])"""
        tr = exl3_parts(prefix)
        suh = st.f32(prefix + ".suh")
        svh = st.f32(prefix + ".svh")
        if nperm is not None:                      # permutation of output Hadamard blocks
            tr = tr[:, (nperm[:, None] * 8 + np.arange(8)).reshape(-1), :]
            svh = svh[(nperm[:, None] * HB + np.arange(HB)).reshape(-1)]
        if kperm is not None:                      # permutation of input Hadamard blocks
            tr = tr[(kperm[:, None] * 8 + np.arange(8)).reshape(-1), :, :]
            suh = suh[(kperm[:, None] * HB + np.arange(HB)).reshape(-1)]
        return np.ascontiguousarray(np.transpose(tr, (1, 0, 2))), suh, svh

    def add_exl3(gname, prefix, nperm=None, kperm=None, n_keep=None):
        tr = exl3_parts(prefix)
        kt, nt, w16 = tr.shape
        if n_keep is not None:   # the head: padded rows past the vocabulary are dropped (whole 16-row tiles)
            assert n_keep % 16 == 0 and n_keep <= nt * 16, (prefix, n_keep, nt)
            nt = n_keep // 16
        bits = w16 // 16
        K, N = kt * 16, nt * 16
        qt = getattr(gguf.GGMLQuantizationType, f"EXL3_{bits}")
        cache = {}

        def get():
            if not cache:
                t, su, sv = exl3_one(prefix, nperm, kperm)
                cache["v"] = (np.ascontiguousarray(t[:nt]), su, sv[:nt * 16]) if n_keep is not None else (t, su, sv)
            return cache["v"]

        def pop(i):
            v = get()[i]
            if i == 2:
                cache.clear()
            return v
        nb = N * K * bits // 8
        plan.append((gname + ".weight", [N, K * bits // 8], np.uint8, nb, qt, lambda: pop(0).view(np.uint8).reshape(N, K * bits // 8)))
        add_f32(gname + ".suh", lambda: pop(1), [K])
        add_f32(gname + ".svh", lambda: pop(2), [N])

    def add_exl3_experts(gname, il, proj):
        pre = [f"{P}layers.{il}.mlp.experts.{e}.{proj}" for e in range(n_expert)]
        tr = exl3_parts(pre[0])
        kt, nt, w16 = tr.shape
        bits = w16 // 16
        K, N = kt * 16, nt * 16
        qt = getattr(gguf.GGMLQuantizationType, f"EXL3_{bits}")
        nb = n_expert * N * K * bits // 8

        def tiles():
            out = np.empty((n_expert, nt, kt, w16), dtype=np.int16)
            for e, p in enumerate(pre):
                t = exl3_parts(p)
                assert t.shape == (kt, nt, w16), (p, t.shape)
                out[e] = np.transpose(t, (1, 0, 2))
            return out.view(np.uint8).reshape(n_expert, N, K * bits // 8)
        plan.append((gname + ".weight", [n_expert, N, K * bits // 8], np.uint8, nb, qt, tiles))
        # routed-expert scales stay fp16 (as stored by exllamav3: lossless) - they live in VRAM next to the GPU ops
        add_f16(gname + ".suh", lambda: np.stack([st.get(p + ".suh")[0] for p in pre]), [n_expert, K])
        add_f16(gname + ".svh", lambda: np.stack([st.get(p + ".svh")[0] for p in pre]), [n_expert, N])

    # embeddings stay bf16 (zero rows up to n_vocab)
    emb, kind = st.get(P + "embed_tokens.weight")
    assert kind == "bf16"
    eshape = [n_vocab, emb.shape[1]]
    def emb_full():
        if emb.shape[0] == n_vocab:
            return np.ascontiguousarray(emb)
        out = np.zeros(eshape, dtype=emb.dtype); out[:emb.shape[0]] = emb
        return out
    plan.append(("token_embd.weight", eshape, np.uint16, n_vocab * emb.shape[1] * 2, gguf.GGMLQuantizationType.BF16, emb_full))

    for il in range(n_layer):
        L = f"{P}layers.{il}."
        b = f"blk.{il}."
        add_f32(b + "attn_norm.weight", lambda L=L: st.f32(L + "input_layernorm.weight") + 1, [tc["hidden_size"]])
        add_f32(b + "post_attention_norm.weight", lambda L=L: st.f32(L + "post_attention_layernorm.weight") + 1, [tc["hidden_size"]])
        lt = tc.get("layer_types")
        is_full = lt[il] == "full_attention" if lt else (il + 1) % full_int == 0
        if is_full:
            assert st.has(L + "self_attn.q_proj.trellis")
            add_exl3(b + "attn_q", L + "self_attn.q_proj")
            add_exl3(b + "attn_k", L + "self_attn.k_proj")
            add_exl3(b + "attn_v", L + "self_attn.v_proj")
            add_exl3(b + "attn_output", L + "self_attn.o_proj")
            add_f32(b + "attn_q_norm.weight", lambda L=L: st.f32(L + "self_attn.q_norm.weight") + 1, [tc["head_dim"]])
            add_f32(b + "attn_k_norm.weight", lambda L=L: st.f32(L + "self_attn.k_norm.weight") + 1, [tc["head_dim"]])
        else:
            A = L + "linear_attn."
            nqk = 2 * nk * hk // HB
            qkv_perm = np.concatenate([np.arange(nqk), nqk + vperm])
            add_exl3(b + "attn_qkv", A + "in_proj_qkv", nperm=qkv_perm)
            add_exl3(b + "attn_gate", A + "in_proj_z", nperm=vperm)
            add_exl3(b + "ssm_out", A + "out_proj", kperm=vperm)
            add_f32(b + "ssm_alpha.weight", lambda A=A: st.f32(A + "in_proj_a.weight")[vperm], [nv, tc["hidden_size"]])
            add_f32(b + "ssm_beta.weight", lambda A=A: st.f32(A + "in_proj_b.weight")[vperm], [nv, tc["hidden_size"]])
            add_f32(b + "ssm_a", lambda A=A: (-np.exp(st.f32(A + "A_log").astype(np.float64))).astype(np.float32)[vperm], [nv])   # float64: correctly rounded (= native loader)
            add_f32(b + "ssm_dt.bias", lambda A=A: st.f32(A + "dt_bias")[vperm], [nv])

            def conv(A=A):
                c = st.f32(A + "conv1d.weight").reshape(-1, tc["linear_conv_kernel_dim"])
                q = 2 * nk * hk
                v = c[q:].reshape(nv, hv, -1)[vperm].reshape(-1, c.shape[1])
                return np.concatenate([c[:q], v], 0)
            add_f32(b + "ssm_conv1d.weight", conv, [2 * nk * hk + nv * hv, tc["linear_conv_kernel_dim"]])
            add_f32(b + "ssm_norm.weight", lambda A=A: st.f32(A + "norm.weight"), [hv])
        M = L + "mlp."
        if not n_expert:   # dense FFN
            add_exl3(b + "ffn_gate", M + "gate_proj")
            add_exl3(b + "ffn_up", M + "up_proj")
            add_exl3(b + "ffn_down", M + "down_proj")
            continue
        add_f32(b + "ffn_gate_inp.weight", lambda M=M: st.f32(M + "gate.weight"), [n_expert, tc["hidden_size"]])
        add_f32(b + "ffn_gate_inp_shexp.weight", lambda M=M: st.f32(M + "shared_expert_gate.weight").reshape(-1), [tc["hidden_size"]])
        add_exl3(b + "ffn_gate_shexp", M + "shared_expert.gate_proj")
        add_exl3(b + "ffn_up_shexp", M + "shared_expert.up_proj")
        add_exl3(b + "ffn_down_shexp", M + "shared_expert.down_proj")
        add_exl3_experts(b + "ffn_gate_exps", il, "gate_proj")
        add_exl3_experts(b + "ffn_up_exps", il, "up_proj")
        add_exl3_experts(b + "ffn_down_exps", il, "down_proj")

    add_f32("output_norm.weight", lambda: st.f32(P + "norm.weight") + 1, [tc["hidden_size"]])
    add_exl3("output", "lm_head")

    for name, shape, dt, nb, raw, _ in plan:
        if raw is None:
            w.add_tensor_info(name, shape, np.dtype(dt), nb)
        elif dt == np.uint16:
            w.add_tensor_info(name, shape, np.dtype(np.float32), nb, raw_dtype=raw)   # bf16: element shape
        else:
            w.add_tensor_info(name, shape, np.dtype(np.uint8), nb, raw_dtype=raw)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    total = 0
    for i, (name, shape, dt, nb, raw, fn) in enumerate(plan):
        a = fn()
        assert a.nbytes == nb, (name, a.nbytes, nb)
        w.write_tensor_data(a)
        total += nb
        if i % 50 == 0:
            print(f"[{i + 1}/{len(plan)}] {name} {total / 2**30:.2f} GiB", flush=True)
    w.close()
    print(f"done: {len(plan)} tensors, {total / 2**30:.2f} GiB -> {out_path}")


if __name__ == "__main__":
    main()

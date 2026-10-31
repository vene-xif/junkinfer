#!/usr/bin/env python3
"""Reader for JDE routing traces (JI_ROUTE_LOG=<dir>, engine jde.cpp jde_rlog_*).

File: header u32[8] = magic 'JIRT', version 1, n_layer, n_expert, n_used, n_embd, flags (bit 0: x logged), 0.
Then per decoded token: i32[4] = tag ('TOKN' with x, 'TOKD' without), pos, token, kv-clear generation;
per layer: u8 ids[n_used] (the router's own top-k), f16 weights[n_used], [f16 x[n_embd] (MoE input) if tag TOKN].
MTP verify steps log draft positions too: a later record at the same (generation, pos) replaces an earlier one.

usage: read_trace.py FILE [--summary] [--npz OUT]   (npz: ids [T, L, k], w [T, L, k], pos, tok, gen [, x [T, L, E]])
"""
import struct, sys
import numpy as np

TOKN, TOKD = 0x4e4b4f54, 0x444b4f54


def read(path, want_x=True):
    with open(path, "rb") as f:
        hdr = struct.unpack("<8I", f.read(32))
        assert hdr[0] == 0x5452494a, "not a JIRT trace"
        nl, ne, nu, nemb = hdr[2], hdr[3], hdr[4], hdr[5]
        recs = {}   # (gen, pos) -> (order, tok, ids, w, x)
        order = 0
        while True:
            b = f.read(16)
            if len(b) < 16:
                break
            tag, pos, tok, gen = struct.unpack("<4i", b)
            tag &= 0xffffffff
            hx = tag == TOKN
            per = nu + 2 * nu + (2 * nemb if hx else 0)
            blob = f.read(per * nl)
            if len(blob) < per * nl:
                break
            a = np.frombuffer(blob, dtype=np.uint8).reshape(nl, per)
            ids = a[:, :nu].copy()
            w = a[:, nu:3 * nu].copy().view(np.float16)
            x = a[:, 3 * nu:].copy().view(np.float16) if (hx and want_x) else None
            recs[(gen, pos)] = (order, tok, ids, w, x)
            order += 1
    keys = sorted(recs, key=lambda k: recs[k][0])   # decode order (last record per position)
    keys.sort(key=lambda k: (k[0], k[1]))           # per conversation generation, by position
    out = {"n_layer": nl, "n_expert": ne, "n_used": nu, "n_embd": nemb,
           "gen": np.array([k[0] for k in keys], np.int32), "pos": np.array([k[1] for k in keys], np.int32),
           "tok": np.array([recs[k][1] for k in keys], np.int32),
           "ids": np.stack([recs[k][2] for k in keys]) if keys else np.zeros((0, nl, nu), np.uint8),
           "w": np.stack([recs[k][3] for k in keys]) if keys else np.zeros((0, nl, nu), np.float16)}
    if want_x and keys and recs[keys[0]][4] is not None and all(recs[k][4] is not None for k in keys):
        out["x"] = np.stack([recs[k][4] for k in keys])
    return out


if __name__ == "__main__":
    path = sys.argv[1]
    d = read(path, want_x="--npz" in sys.argv)
    T = len(d["pos"])
    print(f"{path}: {T} tokens, {len(set(d['gen'].tolist()))} conversation generation(s), layers {d['n_layer']}, "
          f"experts {d['n_expert']} top-{d['n_used']}, x {'yes' if 'x' in d else 'no'}")
    if T > 1:
        ids = d["ids"].astype(np.int32)
        # next-token reuse: fraction of a token's (layer, expert) selections also selected by the previous token
        same = [len(set(map(tuple, np.stack([np.repeat(np.arange(d['n_layer']), d['n_used']), ids[t].ravel()], 1).tolist())) &
                    set(map(tuple, np.stack([np.repeat(np.arange(d['n_layer']), d['n_used']), ids[t - 1].ravel()], 1).tolist())))
                for t in range(1, T)]
        print(f"  previous-token overlap of selections: {np.mean(same) / (d['n_layer'] * d['n_used']):.3f}")
    if "--npz" in sys.argv:
        o = sys.argv[sys.argv.index("--npz") + 1]
        np.savez_compressed(o, **{k: v for k, v in d.items() if isinstance(v, np.ndarray)})
        print("  ->", o)

#!/usr/bin/env python3
"""KL(ref || EXL3 model) on JTOK sets against JREF reference files, same math as junk-ref compare.

usage: exl3_kld.py -m <exl3 dir> --sets S1_wiki.jtok:S1_wiki.jref [...] --summary out.jsonl --label exl3-3.5
"""
import argparse, json, math, os, struct, sys, time

import numpy as np
import torch

sys.path.insert(0, os.environ.get("EXL3_SRC", ""))
from exllamav3 import model_init


def read_jtok(path):
    rows = []
    with open(path, "rb") as f:
        assert f.read(4) == b"JTOK"
        ver, n = struct.unpack("<II", f.read(8))
        assert ver == 1
        for _ in range(n):
            L, a, b = struct.unpack("<III", f.read(12))
            toks = np.frombuffer(f.read(4 * L), dtype="<i4").copy()
            rows.append((toks, a, b))
    return rows


def open_jref(path):
    f = open(path, "rb")
    assert f.read(4) == b"JREF"
    ver, n, K, nv = struct.unpack("<IIII", f.read(16))
    assert ver == 1
    dt = np.dtype([("tlp", "<f4"), ("tail", "<f4"), ("id", "<i4", (K,)), ("lp", "<f4", (K,))])
    return f, n, K, nv, dt


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(ap, cache=False)
    ap.add_argument("--sets", nargs="+", required=True, help="tokens.jtok:ref.jref pairs")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    max_len = 0
    sets = []
    for s in args.sets:
        tp, rp = s.split(":")
        rows = read_jtok(tp)
        max_len = max(max_len, max(len(r[0]) for r in rows))
        sets.append((tp, rp, rows))

    model, config, _, tokenizer = model_init.init(
        args, override_dynamic_seq_len=max(2048, max_len), max_output_size=max(2048, max_len), max_output_factor=5)
    bpw_layer, bpw_head, _ = model.get_storage_info()
    print(f" -- {args.label}: {bpw_layer:.3f} bpw layers / {bpw_head:.2f} bpw head")

    for tp, rp, rows in sets:
        f, n, K, nv, dt = open_jref(rp)
        assert n == len(rows), (rp, n, len(rows))
        kls, nll_m, nll_r, agree, cnt = [], 0.0, 0.0, 0, 0
        t0 = time.time()
        for ri, (toks, a, b) in enumerate(rows):
            (n_pos,) = struct.unpack("<I", f.read(4))
            assert n_pos == b - a, (ri, n_pos, a, b)
            rec = np.fromfile(f, dtype=dt, count=n_pos)
            assert len(rec) == n_pos
            ids = torch.from_numpy(toks[: b + 1].astype(np.int64)).unsqueeze(0)
            logits = model.forward(ids[:, :b], {"attn_mode": "flash_attn_nc"})
            assert logits.shape[-1] >= nv, (logits.shape, nv)
            for c0 in range(0, n_pos, args.chunk):
                c1 = min(n_pos, c0 + args.chunk)
                lg = logits[0, a + c0:a + c1, :nv].float()
                lq = torch.log_softmax(lg, dim=-1)
                rid = torch.from_numpy(rec["id"][c0:c1].astype(np.int64)).to(lq.device)
                rlp = torch.from_numpy(rec["lp"][c0:c1]).to(lq.device).double()
                tail = torch.from_numpy(rec["tail"][c0:c1]).to(lq.device).double()
                lqk = lq.gather(1, rid).double()
                kl = (rlp.exp() * (rlp - lqk)).sum(1)
                q_tail = (1.0 - lqk.exp().sum(1)).clamp_min(1e-30)
                kl = kl + torch.where(tail > 0, tail * (tail.clamp_min(1e-300).log() - q_tail.log()), torch.zeros_like(tail))
                kls.append(kl.float().cpu())
                tgt = torch.from_numpy(toks[a + c0 + 1:a + c1 + 1].astype(np.int64)).to(lq.device)
                nll_m -= lq.gather(1, tgt.unsqueeze(1)).sum().item()
                nll_r -= float(rec["tlp"][c0:c1].astype(np.float64).sum())
                agree += (lq.argmax(1) == rid[:, 0]).sum().item()
                cnt += c1 - c0
            del logits
            torch.cuda.empty_cache()
        extra = f.read(1)
        assert not extra, "JREF trailing data"
        f.close()
        allkl = torch.cat(kls).numpy()
        s = np.sort(allkl)
        q = lambda p: float(s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))])
        res = {"label": args.label, "tokens": tp, "n": cnt, "kl_mean": float(allkl.mean()), "kl_p50": q(0.5),
               "kl_p90": q(0.9), "kl_p99": q(0.99), "kl_p999": q(0.999), "kl_max": float(s[-1]),
               "top1_agree": agree / cnt, "ppl_model": math.exp(nll_m / cnt), "ppl_ref": math.exp(nll_r / cnt),
               "seconds": round(time.time() - t0, 1), "bpw_layers": bpw_layer, "bpw_head": bpw_head}
        print("SUMMARY " + json.dumps(res), flush=True)
        with open(args.summary, "a") as fo:
            fo.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

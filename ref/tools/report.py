#!/usr/bin/env python3
"""Summarize kld.jsonl into a markdown table: one row per (label, set), last entry wins."""
import json, os, sys

kld, out, data = sys.argv[1], sys.argv[2], sys.argv[3]
rows = {}
if os.path.exists(kld):
    for line in open(kld):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = os.path.basename(r.get("tokens", "?")).replace(".jtok", "")
        rows[(r["label"], s)] = r

sets = sorted({s for _, s in rows})
lines = ["# JunkInfer reference run - KLD vs BF16", "",
         "KL(ref || model) over the BF16 top-512 + tail bucket. Lower is better.", ""]
lines.append("| label | set | n | KL mean | KL p99 | top-1 agree | PPL model | PPL ref |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
for (label, s) in sorted(rows, key=lambda k: (k[0], k[1])):
    r = rows[(label, s)]
    lines.append(f"| {label} | {s} | {r['n']} | {r['kl_mean']:.5f} | {r['kl_p99']:.4f} | {100 * r['top1_agree']:.2f}% | "
                 f"{r['ppl_model']:.4f} | {r['ppl_ref']:.4f} |")

# mean over the three short sets, for a one-number ranking
lines += ["", "## Mean KL over S1+S2+S3 (token-weighted)", "", "| label | KL mean | top-1 |", "|---|---:|---:|"]
by = {}
for (label, s), r in rows.items():
    if s.startswith("S"):
        b = by.setdefault(label, [0.0, 0.0, 0])
        b[0] += r["kl_mean"] * r["n"]; b[1] += r["top1_agree"] * r["n"]; b[2] += r["n"]
for label, (k, t, n) in sorted(by.items(), key=lambda kv: kv[1][0] / max(kv[1][2], 1)):
    lines.append(f"| {label} | {k / n:.5f} | {100 * t / n:.2f}% |")

ev = os.path.dirname(out)
for f in sorted(os.listdir(ev)):
    if f.endswith(".txt") and ("ppl" in f):
        txt = open(os.path.join(ev, f), errors="replace").read().splitlines()
        keep = [l for l in txt if "Mean" in l and "KLD" in l or "Same top" in l]
        lines += ["", f"## {f}", "```"] + keep[-4:] + ["```"]

open(out, "w").write("\n".join(lines) + "\n")
print(f"report: {len(rows)} entries -> {out}")

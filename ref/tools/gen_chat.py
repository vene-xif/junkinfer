#!/usr/bin/env python3
"""Generate the in-distribution chat set (S3) with the reference model itself.

Each prompt is rendered with the model's own HF chat template (tools included), tokenized with the
HF tokenizer, and sent as token ids to a running llama-server (/completion, return_tokens). The full
sequence (prompt + generated tokens) becomes one JTOK row scored on the generated part, so the KLD
gate measures the model on the text it actually writes (thinking, code, tool calls, other languages).

Outputs in --out:
  S3_chat.jtok        eval split rows
  calib_chat.txt      calib split text (prompt + reply), appended to the imatrix calibration
  chat_samples.json   every prompt and reply as text, for human review
"""
import argparse, json, os, struct, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

MAX_SEQ = 8000


def post(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def write_jtok(path, rows):
    with open(path, "wb") as f:
        f.write(b"JTOK")
        f.write(struct.pack("<II", 1, len(rows)))
        for toks, a, b in rows:
            assert 0 <= a < b <= len(toks) - 1, (a, b, len(toks))
            f.write(struct.pack("<III", len(toks), a, b))
            f.write(struct.pack(f"<{len(toks)}i", *toks))


def render(tokenizer, spec):
    kw = {}
    if spec.get("tools"):
        kw["tools"] = spec["tools"]
    text = tokenizer.apply_chat_template(spec["messages"], tokenize=False, add_generation_prompt=True, **kw)
    if spec.get("tools"):
        for t in spec["tools"]:
            name = t["function"]["name"]
            if name not in text:
                raise RuntimeError(f"{spec['id']}: tool {name} missing from rendered prompt")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:18950")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--check-only", action="store_true", help="render every prompt, no server, write nothing")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir)
    with open(args.prompts) as f:
        specs = json.load(f)
    ids_seen = set()
    rendered = []
    for s in specs:
        assert s["id"] not in ids_seen, s["id"]
        ids_seen.add(s["id"])
        assert s["split"] in ("eval", "calib"), s["id"]
        text, ids = render(tokenizer, s)
        assert len(ids) + s.get("max_tokens", 768) <= MAX_SEQ, (s["id"], len(ids))
        rendered.append((s, text, ids))
    n_eval = sum(1 for s in specs if s["split"] == "eval")
    print(f"gen_chat: {len(specs)} prompts render OK ({n_eval} eval), prompt tokens "
          f"min {min(len(r[2]) for r in rendered)} max {max(len(r[2]) for r in rendered)}")
    if args.check_only:
        return

    os.makedirs(args.out, exist_ok=True)

    def run(i):
        s, text, ids = rendered[i]
        t0 = time.time()
        r = post(args.server + "/completion", {
            "prompt": ids, "n_predict": s.get("max_tokens", 768),
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "seed": 1000 + i, "return_tokens": True, "cache_prompt": False, "stream": False,
        })
        gen = r.get("tokens") or []
        dt = time.time() - t0
        print(f"  {s['id']}: prompt {len(ids)} gen {len(gen)} in {dt:.0f}s ({len(gen) / max(dt, 1e-9):.1f} tok/s)", flush=True)
        return s, text, ids, gen, r.get("content", "")

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        results = list(ex.map(run, range(len(rendered))))

    rows, samples, calib = [], [], []
    for s, text, ids, gen, content in results:
        samples.append({"id": s["id"], "split": s["split"], "prompt_text": text, "reply_text": content,
                        "prompt_tokens": len(ids), "reply_tokens": len(gen)})
        if len(gen) < 8:
            print(f"  WARNING {s['id']}: only {len(gen)} generated tokens, skipped", file=sys.stderr)
            continue
        seq = list(ids) + list(gen)
        if s["split"] == "eval":
            rows.append((seq, len(ids) - 1, len(seq) - 1))
        else:
            calib.append(text + content)
    if len(rows) < max(4, n_eval // 2):
        sys.exit(f"gen_chat: too few usable eval rows ({len(rows)})")
    write_jtok(os.path.join(args.out, "S3_chat.jtok"), rows)
    with open(os.path.join(args.out, "calib_chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(calib))
    with open(os.path.join(args.out, "chat_samples.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)
    print(f"gen_chat: wrote {len(rows)} eval rows, {sum(r[2] - r[1] for r in rows)} scored tokens, {len(calib)} calib texts")


if __name__ == "__main__":
    main()

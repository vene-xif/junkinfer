#!/usr/bin/env python3
"""Decode speed at depth through the OpenAI server: a long prompt (llama.cpp docs, ~4 chars/token), then a short
generation. Reports prompt tokens, prefill t/s and decode t/s from the server's timings.

usage: depth_bench.py [approx_prompt_tokens=30000] [gen_tokens=128] [base_url]
"""
import glob, json, sys, urllib.request

n_tok = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 128
base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080"

text = ""
for p in sorted(glob.glob("engine/docs/**/*.md", recursive=True)):
    text += open(p, errors="ignore").read() + "\n\n"
    if len(text) > n_tok * 4:
        break
text = text[: n_tok * 4]
body = {"messages": [{"role": "user", "content": text + "\n\nSummarise the documents above in a few paragraphs."}],
        "max_tokens": n_gen, "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=7200) as r:
    t = json.loads(r.read())["timings"]
print(f"prompt {t['prompt_n']} tokens: prefill {t['prompt_per_second']:.1f} t/s ({t['prompt_ms']/1000:.0f} s); "
      f"decode {t['predicted_n']} tokens at {t['predicted_per_second']:.1f} t/s")

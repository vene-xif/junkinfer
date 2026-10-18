#!/usr/bin/env python3
"""Decode speed at depth through the OpenAI server: a long prompt (llama.cpp docs, ~4 chars/token), then a short
generation. Reports prompt tokens, prefill t/s and decode t/s from the server's timings.

usage: depth_bench.py [approx_prompt_tokens=30000] [gen_tokens=128] [base_url] [show] [nocache]  (show: print the reply; nocache: no prompt-cache reuse)
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
        "max_tokens": n_gen, "temperature": float(__import__("os").environ.get("JI_TEMP", "0.7")), "chat_template_kwargs": {"enable_thinking": False}}
if len(sys.argv) > 5 and sys.argv[5] == "nocache":
    body["cache_prompt"] = False
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=7200) as r:
    d = json.loads(r.read())
    t = d["timings"]
    if len(sys.argv) > 4: print(repr(d["choices"][0]["message"]["content"][:200]))
print(f"prompt {t['prompt_n']} tokens: prefill {t['prompt_per_second']:.1f} t/s ({t['prompt_ms']/1000:.0f} s); "
      f"decode {t['predicted_n']} tokens at {t['predicted_per_second']:.1f} t/s")

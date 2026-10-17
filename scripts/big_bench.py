#!/usr/bin/env python3
"""Prefill/decode at near-full context: concatenates source files of the engine (C/C++ and docs) until ~N tokens
(~3.3 chars/token for code), asks for a short summary, prints the server timings."""
import glob, json, sys, urllib.request
n_tok = int(sys.argv[1]) if len(sys.argv) > 1 else 250000
text = ""
for pat in ["engine/docs/**/*.md", "engine/src/**/*.cpp", "engine/ggml/src/ggml-vulkan/*.cpp", "engine/common/*.cpp", "engine/tools/**/*.cpp"]:
    for p in sorted(glob.glob(pat, recursive=True)):
        text += open(p, errors="ignore").read() + "\n\n"
        if len(text) > n_tok * 3.3: break
    if len(text) > n_tok * 3.3: break
text = text[: int(n_tok * 3.3)]
body = {"messages": [{"role": "user", "content": text + "\n\nIn two sentences, what is this code base?"}],
        "max_tokens": 64, "temperature": 0.0, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=10800) as r:
    d = json.loads(r.read()); t = d["timings"]
    print(repr(d["choices"][0]["message"]["content"][:200]))
    print(f"prompt {t['prompt_n']} tokens: prefill {t['prompt_per_second']:.1f} t/s ({t['prompt_ms']/1000:.0f} s); decode {t['predicted_n']} tokens at {t['predicted_per_second']:.1f} t/s")

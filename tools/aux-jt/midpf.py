#!/usr/bin/env python3
"""Prefill time of N new tokens after a cached prefix (raw /completion, cache on): usage midpf.py N [N ...]"""
import json, sys, urllib.request, os
U = "http://127.0.0.1:8080"
base = open(os.path.expanduser("~/jt/p8k.txt")).read()[:6000]
extra = open(os.path.expanduser("~/jt/p68k.txt")).read()
def run(prompt):
    req = urllib.request.Request(U + "/completion", data=json.dumps({"prompt": prompt, "n_predict": 1, "temperature": 0, "cache_prompt": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["timings"]
for n in [int(a) for a in sys.argv[1:]]:
    run(base)                                     # prefix cached
    t = run(base + extra[: int(n * 3.6)])
    print(f"+{t['prompt_n']} tokens: {t['prompt_ms']:.0f} ms ({t['prompt_per_second']:.0f} t/s)")

#!/usr/bin/env python3
"""Prefill speed at depth from saved KV: restores /mnt/ssd/kvcache/depth_N.bin into slot 0, then sends the same
prompt with ~EXTRA new tokens inserted after the saved text, so only the tail is prefilled at depth N.
usage: dpb.py N [extra_tokens=4096]   (server started with JI_SLOT_DIR=/mnt/ssd/kvcache)"""
import glob, json, os, sys, urllib.request
N = int(sys.argv[1]); extra = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
U = "http://127.0.0.1:8080"
def post(path, body, t=7200):
    r = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))
os.chdir(os.path.expanduser("~/junkinfer"))
if N > 110000:
    text = open(f"/mnt/ssd/kvcache/depth_{N}.txt").read(); q = "\n\nIn two sentences, what is this code base?"
else:
    text = ""
    for p in sorted(glob.glob("engine/docs/**/*.md", recursive=True)):
        text += open(p, errors="ignore").read() + "\n\n"
        if len(text) > N * 4: break
    text = text[: N * 4]; q = "\n\nSummarise the documents above in a few paragraphs."
tail = open(os.path.expanduser("~/jt/p68k.txt")).read()[: int(extra * 3.5)]
r = post("/slots/0?action=restore", {"filename": f"depth_{N}.bin"})
body = {"messages": [{"role": "user", "content": text + "\n\n" + tail + q}], "max_tokens": 4, "temperature": 0.0,
        "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
d = post("/v1/chat/completions", body); t = d["timings"]
print(f"depth {r.get('n_restored', '?')}: +{t['prompt_n']} tokens prefill {t['prompt_per_second']:.1f} t/s ({t['prompt_ms']/1e3:.1f} s)")

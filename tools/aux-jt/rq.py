#!/usr/bin/env python3
# restore depth_N.bin, then a greedy question appended to the saved text: prints t/s + answer md5 + head
import json, sys, hashlib, urllib.request
N = sys.argv[1]; U = "http://127.0.0.1:8080"
def post(p, b):
    r = urllib.request.Request(U + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
r = post("/slots/0?action=restore", {"filename": f"depth_{N}.bin"})
d = post("/completion", {"prompt": "", "n_predict": 0, "cache_prompt": True, "id_slot": 0}) if False else None
toks = post("/slots/0?action=save", {"filename": "rq_tmp.bin"}) if False else None
# continue the restored tokens: /completion with the slot prompt reuse needs the text; use the frozen prompt file
import os
txt = open(f"/mnt/ssd/kvcache/depth_{N}.txt").read() if os.path.exists(f"/mnt/ssd/kvcache/depth_{N}.txt") else None
print("restored", r.get("n_restored"))

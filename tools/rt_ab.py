#!/usr/bin/env python3
"""In-instance A/B: alternate runtime-switch configs (~/junkinfer/runtime.env) between requests on the running
server. usage: rt_ab.py "KEY=V KEY=V" "KEY=V" [rounds=2] [max_tokens=1100]
Each round runs every prompt under every config (A B A B ...); prints t/s per request, means, and whether the greedy
outputs of the same prompt match across configs (an exactness signal, not a proof)."""
import json, os, sys, time, hashlib, urllib.request, collections
RT = os.path.expanduser("~/junkinfer/runtime.env")
P = ["Explain in detail how a modern CPU executes instructions: pipelining, branch prediction, out-of-order execution, caches. Be thorough.",
     "Write a long story about a lighthouse keeper who discovers a strange signal in the fog.",
     "Write a Python implementation of a red-black tree with insertion and deletion, and explain each rotation case.",
     "Compare the economic policies of the Roman Republic and the early Roman Empire, with examples."]
cfgs = [c.split() for c in sys.argv[1:3]]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ntok = int(sys.argv[4]) if len(sys.argv) > 4 else 1100


def setcfg(c):
    with open(RT + ".tmp", "w") as f:
        f.write("\n".join(c) + "\n")
    os.replace(RT + ".tmp", RT)
    time.sleep(0.6)   # the engine checks the file at most every 250 ms, at a decode step


def req(p, n=None):
    d = json.dumps({"messages": [{"role": "user", "content": p}], "max_tokens": n or ntok, "temperature": 0, "ignore_eos": True}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", d,
                                                                {"Content-Type": "application/json"}), timeout=1200))
    m = r["choices"][0]["message"]
    txt = (m.get("reasoning_content") or "") + (m.get("content") or "")
    return r["timings"]["predicted_per_second"], hashlib.md5(txt.encode()).hexdigest()[:10]


res = collections.defaultdict(list)
md = collections.defaultdict(set)
for rnd in range(rounds):
    for i, p in enumerate(P):
        for ci, c in enumerate(cfgs):
            setcfg(c)
            req("ok", 8)   # a short request: the switch is applied (+ re-recorded) before the timed one
            tps, h = req(p)
            res[ci].append(tps)
            md[i].add(h)
            name = " ".join(c) or "(defaults)"
            print(f"r{rnd} p{i} cfg{ci} {name}: {tps:.2f} t/s {h}", flush=True)
for ci, c in enumerate(cfgs):
    v = res[ci]
    print(f"MEAN cfg{ci} {' '.join(c) or '(defaults)'}: {sum(v) / len(v):.2f} t/s over {len(v)}")
print("outputs identical across configs per prompt:", {i: len(s) == 1 for i, s in md.items()})
os.remove(RT)

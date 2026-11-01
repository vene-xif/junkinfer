#!/usr/bin/env python3
"""In-instance prefill A/B: alternate runtime-switch configs (~/junkinfer/runtime.env) between ~10k-token prompts
(cache_prompt false, 1 output token). usage: rt_pf.py "CFG" "CFG" ... [--rounds N]  (CFG: "KEY=V KEY=V", "" = defaults)"""
import json, os, sys, time, urllib.request, collections
RT = os.path.expanduser("~/junkinfer/runtime.env")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 2
if "--rounds" in sys.argv:
    args = [a for a in args if a != sys.argv[sys.argv.index("--rounds") + 1]]
txt = open(os.path.expanduser("~/jt/pf_corpus.md")).read()
chunks = [txt[i * 30000:i * 30000 + 26000] for i in range(3)]


def setcfg(c):
    with open(RT + ".tmp", "w") as f:
        f.write("\n".join(c.split()) + "\n")
    os.replace(RT + ".tmp", RT)
    time.sleep(0.6)


def pf(chunk):
    d = json.dumps({"messages": [{"role": "user", "content": "Summarize:\n" + chunk}], "max_tokens": 1, "cache_prompt": False}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", d, {"Content-Type": "application/json"}), timeout=1200))
    return r["timings"]["prompt_n"], r["timings"]["prompt_per_second"]


res = collections.defaultdict(list)
setcfg(args[0]); pf(chunks[0][:2000])   # warm-up (first prefill after a start is slower)
for rnd in range(rounds):
    for ci, c in enumerate(args):
        setcfg(c)
        for k, ch in enumerate(chunks):
            n, tps = pf(ch)
            res[ci].append(tps)
            print(f"r{rnd} cfg{ci} [{c or 'defaults'}] prompt{k} {n} tok: {tps:.1f} t/s", flush=True)
for ci, c in enumerate(args):
    v = res[ci]
    print(f"MEAN cfg{ci} [{c or 'defaults'}]: {sum(v) / len(v):.1f} t/s over {len(v)}")
os.remove(RT)

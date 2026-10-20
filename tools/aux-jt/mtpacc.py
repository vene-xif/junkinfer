import json, sys, urllib.request
# MTP draft acceptance over a fixed prompt set: temperature 0 and 0.7 (fixed seed); prints accepted / drafted
code = open("/home/vene/junkinfer/scripts/smoke_api.py").read()
prompts = [
    "Rewrite the following Python script with type hints added to every function. Output the complete file only.\n\n```python\n" + code + "\n```",
    "Tell me a short story about a lighthouse keeper.",
    "Explain how a CPU cache works, including cache lines, associativity and write-back policies.",
    "Write a bash script that finds the 10 largest files under a directory and prints their sizes in human-readable form.",
    "Napisz krótki wiersz o jesieni w Krakowie.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the journey? Show the calculation step by step.",
]
for temp in (0.0, 0.7):
    dn = da = pn = 0; pms = 0.0
    for p in prompts:
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": 300, "temperature": temp, "seed": 42,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t = json.load(urllib.request.urlopen(req, timeout=3600))["timings"]
        dn += t.get("draft_n", 0); da += t.get("draft_n_accepted", 0); pn += t["predicted_n"]; pms += t["predicted_ms"]
    print("%s temp %.1f: drafted %d accepted %d rate %.3f (%d tokens) decode %.1f t/s" % (sys.argv[1], temp, dn, da, da / max(1, dn), pn, pn / (pms / 1e3)), flush=True)

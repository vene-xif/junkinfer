#!/usr/bin/env python3
"""Engine comparison client (stock llama.cpp vs JunkInfer), same requests for every server.

Short context: JunkInfer's three spectest prompts (rewrite / story / explain), 3072 tokens each, greedy, ignore_eos.
Depth: prompts of ~8k / ~16k / ~32k tokens cut from a corpus (token counts checked with the server's /tokenize),
cache_prompt off (full prefill every time), then 3072 generated tokens at that depth.
Usage: cmp_bench.py LABEL corpus.txt [url] ; appends one line per request to cmp_results.tsv
"""
import json, sys, time, urllib.request

label, corpus = sys.argv[1], sys.argv[2]
url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080"

def post(path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=7200))

def ntok(text):
    return len(post("/tokenize", {"content": text})["tokens"])

def run(name, content):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 3072, "ignore_eos": True, "temperature": 0,
            "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    d = post("/v1/chat/completions", body)
    t = d["timings"]
    line = "%s\t%s\t%d\t%.1f\t%d\t%.2f\t%s\t%s\t%.0f" % (label, name, t["prompt_n"], t["prompt_per_second"], t["predicted_n"],
           t["predicted_per_second"], t.get("draft_n", "-"), t.get("draft_n_accepted", "-"), time.time() - t0)
    print(line, flush=True)
    open("cmp_results.tsv", "a").write(line + "\n")

code = open("/home/vene/junkinfer/scripts/smoke_api.py").read()
short = {
    "rewrite": "Rewrite the following Python script with type hints added to every function. Output the complete file only, no explanation.\n\n```python\n" + code + "\n```",
    "story": "Tell me a short story about a lighthouse keeper.",
    "explain": "Explain how a hash map works internally, including collision handling and resizing, with a small C example.",
}
for n, p in short.items():
    run(n, p)

text = open(corpus, encoding="utf-8", errors="ignore").read()
for target in (8192, 16384, 32000):
    lo, hi = 1000, len(text)
    while hi - lo > 2000:                      # chars giving ~target tokens
        mid = (lo + hi) // 2
        if ntok(text[:mid]) < target: lo = mid
        else: hi = mid
    run("depth%dk" % (target // 1000), text[:lo] + "\n\nSummarize the text above in detail, section by section.")

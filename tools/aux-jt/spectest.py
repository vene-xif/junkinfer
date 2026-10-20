import json, sys, urllib.request
# decode t/s on two prompts: a code rewrite (copy-heavy) and a story (novel text); greedy
code = open("/home/vene/junkinfer/scripts/smoke_api.py").read()
prompts = {
  "rewrite": "Rewrite the following Python script with type hints added to every function. Output the complete file only, no explanation.\n\n```python\n" + code + "\n```",
  "story": "Tell me a short story about a lighthouse keeper.",
}
for name, p in prompts.items():
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 600, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    t = d["timings"]
    extra = " draft %s accepted %s" % (t.get("draft_n", "-"), t.get("draft_n_accepted", "-"))
    print("%-8s %d tokens %.1f t/s%s | %r" % (name, t["predicted_n"], t["predicted_per_second"], extra, d["choices"][0]["message"]["content"][:60]))

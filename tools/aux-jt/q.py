import json, sys, urllib.request
body = {"messages": [{"role": "user", "content": sys.argv[1] if len(sys.argv) > 1 else "Tell me a short story about a lighthouse keeper."}], "max_tokens": int(sys.argv[2]) if len(sys.argv) > 2 else 40, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=3600)); t = d["timings"]
print("%.1f t/s | %r" % (t["predicted_per_second"], d["choices"][0]["message"]["content"][:100]))

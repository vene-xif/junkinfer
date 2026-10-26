import json, urllib.request, hashlib
msgs=[{"role":"user","content":"Write a short poem about the sea."}]
for turn in range(3):
    req=urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps({"messages":msgs,"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode(), {"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(req)); c=d["choices"][0]["message"]["content"]
    print(turn, round(d["timings"]["predicted_per_second"],1), hashlib.md5(c.encode()).hexdigest()[:8], repr(c[:40]))
    msgs += [{"role":"assistant","content":c},{"role":"user","content":"Now another one, about mountains." if turn==0 else "And one about the city."}]

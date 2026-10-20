import json, sys, urllib.request
# server prefill of a prompt file (raw completion endpoint, cache off): prints prompt t/s and decode t/s
txt = open(sys.argv[1]).read()
req = urllib.request.Request("http://127.0.0.1:8080/completion", data=json.dumps({"prompt": txt, "n_predict": 16, "temperature": 0, "cache_prompt": False}).encode(), headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=7200))
t = d["timings"]
print(f"prompt {t['prompt_n']} tokens: {t['prompt_per_second']:.1f} t/s ({t['prompt_ms']/1e3:.0f} s); decode {t['predicted_per_second']:.1f} t/s")

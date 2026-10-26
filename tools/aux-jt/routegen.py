import json, sys, urllib.request
prompts = [
    "Write a Python function that parses a CSV file and computes per-column statistics, with tests.",
    "Write a long story about a lighthouse keeper who discovers a message in a bottle.",
    "Explain how TCP congestion control works, including slow start, AIMD and fast recovery.",
    "Write a bash script that backs up a directory with rotation and logs errors.",
    "Napisz wiersz o jesieni w Krakowie.",
    "Solve step by step: a train leaves at 9:40 going 84 km/h, another at 10:15 going 112 km/h; when does the second catch up?",
]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
for p in prompts:
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": p}],
        "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    print(round(d["timings"]["predicted_per_second"], 1), d["usage"]["completion_tokens"], flush=True)

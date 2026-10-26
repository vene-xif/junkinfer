#!/usr/bin/env python3
"""Turn-to-turn latency through the OpenAI API: a chat with a long system prompt and short user turns, streamed.
Per turn: time to first token, prompt tokens processed vs cached (server timings), prefill ms, decode t/s.
usage: turns.py [n_turns=5] [system_tokens~2000] [max_tokens=60] [think=0]"""
import json, sys, time, urllib.request, os
U = os.environ.get("URL", "http://127.0.0.1:8080")
nt = int(sys.argv[1]) if len(sys.argv) > 1 else 5
st = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
mt = int(sys.argv[3]) if len(sys.argv) > 3 else 60
think = len(sys.argv) > 4 and sys.argv[4] == "1"
sysmsg = open(os.path.expanduser("~/jt/p8k.txt")).read()[: st * 4]
msgs = [{"role": "system", "content": "You are a helpful assistant. Reference material follows.\n\n" + sysmsg}]
qs = ["Summarise the reference material in two sentences.", "What is the most important detail in it?",
      "Name one weakness of the approach described.", "Suggest an improvement.", "Thanks. One more sentence of advice?",
      "Rephrase that more briefly.", "And in five words?", "Good. Now say goodbye."]
for i in range(nt):
    msgs.append({"role": "user", "content": qs[i % len(qs)]})
    body = {"messages": msgs, "max_tokens": mt, "temperature": 0.0, "stream": True, "chat_template_kwargs": {"enable_thinking": think},
            "stream_options": {"include_usage": True}, "timings_per_token": False}
    req = urllib.request.Request(U + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; tim = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("timings"): tim = d["timings"]
            for c in d.get("choices", []):
                delta = c.get("delta", {}).get("content") or ""
                if delta and ttft is None: ttft = time.time() - t0
                text += delta
    tot = time.time() - t0
    msgs.append({"role": "assistant", "content": text})
    t = tim or {}
    print(f"turn {i+1}: ttft {ttft*1000 if ttft else -1:.0f} ms | prompt {t.get('prompt_n','?')} new, {t.get('cache_n','?')} cached, "
          f"{t.get('prompt_ms',0):.0f} ms | decode {t.get('predicted_n','?')} tok at {t.get('predicted_per_second',0):.1f} t/s | total {tot:.1f} s")

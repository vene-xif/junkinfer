#!/usr/bin/env python3
"""MVP API smoke test: OpenAI chat, chat template, tool-call parsing, prefix caching.

usage: smoke_api.py [base_url]     (default http://127.0.0.1:8080)
Exits non-zero on the first failed check. Short: a few hundred generated tokens in total.
"""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        sys.exit(1)


# 1. plain chat through the chat template (thinking off to keep it short)
r = post("/v1/chat/completions", {
    "messages": [{"role": "user", "content": "Reply with exactly one word: the capital of France."}],
    "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
msg = r["choices"][0]["message"]["content"] or ""
check("chat completion", "paris" in msg.lower(), repr(msg[:60]))
t = r.get("timings", {})
print(f"      decode {t.get('predicted_per_second', 0):.1f} t/s")

# 2. tool call parsed into OpenAI tool_calls
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = post("/v1/chat/completions", {
    "messages": [{"role": "user", "content": "What is the weather in Krakow right now? Use the tool."}],
    "tools": tools, "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
m = r["choices"][0]["message"]
calls = m.get("tool_calls") or []
ok = bool(calls) and calls[0]["function"]["name"] == "get_weather" and "krak" in json.loads(calls[0]["function"]["arguments"]).get("city", "").lower()
check("tool call parsed", ok, json.dumps(calls)[:120])
check("finish_reason tool_calls", r["choices"][0]["finish_reason"] == "tool_calls", r["choices"][0]["finish_reason"])

# 3. prefix caching: a long shared prefix, second request must reuse it
doc = " ".join(f"Item {i}: the value of field {i % 7} is {i * 37 % 101}." for i in range(400))
base = [{"role": "system", "content": "You answer questions about the document.\n\n" + doc}]
q1 = base + [{"role": "user", "content": "What is the value of item 12?"}]
q2 = base + [{"role": "user", "content": "What is the value of item 345?"}]
t0 = time.time(); r1 = post("/v1/chat/completions", {"messages": q1, "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}); d1 = time.time() - t0
t0 = time.time(); r2 = post("/v1/chat/completions", {"messages": q2, "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}); d2 = time.time() - t0
p1, p2 = r1["timings"]["prompt_n"], r2["timings"]["prompt_n"]
cached = r2["timings"].get("cache_n", 0)
check("prefix cache reused", p2 < p1 / 4 and cached > p1 / 2, f"prompt_n {p1} -> {p2}, cache_n {cached}, wall {d1:.1f}s -> {d2:.1f}s")
print(f"      prefill {r1['timings']['prompt_per_second']:.1f} t/s on {p1} tokens")
print("ALL PASS")

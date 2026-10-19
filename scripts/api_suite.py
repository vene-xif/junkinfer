#!/usr/bin/env python3
"""JunkInfer MVP API suite: what real clients (OpenAI / Anthropic SDKs, coding agents, chat UIs) rely on.

usage: api_suite.py [base_url] [-k substring] [--long] [-v]     (default http://127.0.0.1:8080; needs `pip install openai anthropic`)
Runs every check, prints PASS / FAIL with details, exits non-zero if any failed. Takes several minutes (thinking
and tool-loop checks generate a few thousand tokens in total).
"""
import json, sys, threading, time, traceback, urllib.request

import anthropic
import openai

args = [a for a in sys.argv[1:]]
SEL = None
if "-k" in args:
    i = args.index("-k"); SEL = args[i + 1]; del args[i:i + 2]
args = [a for a in args if not a.startswith("-")]   # --long, -v
BASE = args[0] if args else "http://127.0.0.1:8080"
oc = openai.OpenAI(base_url=BASE + "/v1", api_key="none", timeout=1800)
ac = anthropic.Anthropic(base_url=BASE, api_key="none", timeout=1800)
MODEL = "junkinfer"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
results = []
DOC = " ".join(f"Item {i}: the value of field {i % 7} is {i * 37 % 101}." for i in range(400))


def test(fn):
    name = fn.__name__
    if SEL and SEL not in name:
        return fn
    t0 = time.time()
    try:
        detail = fn() or ""
        results.append((name, True, detail))
        print(f"PASS  {name:34s} {time.time() - t0:5.1f}s  {detail}", flush=True)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        results.append((name, False, msg))
        print(f"FAIL  {name:34s} {time.time() - t0:5.1f}s  {msg[:400]}", flush=True)
        if "-v" in sys.argv:
            traceback.print_exc()
    return fn


def timings(r):
    return (r.model_extra or {}).get("timings", {})


def need(cond, msg):
    if not cond:
        raise AssertionError(msg)


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
TIME = {"type": "function", "function": {
    "name": "get_time", "description": "Get the current local time in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
BOOK = {"type": "function", "function": {
    "name": "create_booking", "description": "Create a restaurant booking.",
    "parameters": {"type": "object", "properties": {
        "restaurant": {"type": "string"},
        "party_size": {"type": "integer"},
        "outdoor": {"type": "boolean"},
        "dishes": {"type": "array", "items": {"type": "string"}},
        "contact": {"type": "object", "properties": {"name": {"type": "string"}, "phone": {"type": "string"}},
                    "required": ["name", "phone"]}},
        "required": ["restaurant", "party_size", "outdoor", "dishes", "contact"]}}}


def chat(messages, **kw):
    return oc.chat.completions.create(model=MODEL, messages=messages, **kw)


# ---------------------------------------------------------------- basics

@test
def models_list():
    ms = oc.models.list().data
    need(len(ms) >= 1, "no models")
    return ms[0].id


@test
def chat_basic():
    r = chat([{"role": "user", "content": "Reply with exactly one word: the capital of France."}],
             max_tokens=32, temperature=0, extra_body=NOTHINK)
    c = r.choices[0].message.content or ""
    need("paris" in c.lower(), repr(c))
    need(r.choices[0].finish_reason == "stop", r.choices[0].finish_reason)
    need(r.usage and r.usage.prompt_tokens > 0 and r.usage.completion_tokens > 0, f"usage {r.usage}")
    return repr(c)


@test
def chat_thinking_default():
    # no chat_template_kwargs: the model thinks; reasoning must come back separately, not inside content
    r = chat([{"role": "user", "content": "What is 17 * 23? Answer with the number only."}], max_tokens=2048, temperature=0)
    m = r.choices[0].message
    c = m.content or ""
    rc = getattr(m, "reasoning_content", None) or (m.model_extra or {}).get("reasoning_content")
    need("391" in c, f"content {c!r}")
    need("<think>" not in c and "</think>" not in c, f"think tags in content {c[:80]!r}")
    need(rc and len(rc) > 10, f"reasoning_content missing ({rc!r})")
    return f"content {c.strip()!r}, reasoning {len(rc)} chars"


@test
def chat_stream_usage():
    s = chat([{"role": "user", "content": "Count from 1 to 10 separated by spaces."}], max_tokens=64, temperature=0,
             stream=True, stream_options={"include_usage": True}, extra_body=NOTHINK)
    text, fin, usage, n = "", None, None, 0
    for ch in s:
        n += 1
        if ch.usage:
            usage = ch.usage
        for c in ch.choices:
            text += c.delta.content or ""
            fin = c.finish_reason or fin
    need("1 2 3 4 5 6 7 8 9 10" in text, repr(text))
    need(fin == "stop", f"finish {fin}")
    need(usage and usage.completion_tokens > 0, f"usage {usage}")
    return f"{n} chunks, {usage.completion_tokens} tokens"


@test
def chat_stream_thinking():
    s = chat([{"role": "user", "content": "Is 91 a prime number? Answer yes or no."}], max_tokens=2048, temperature=0, stream=True)
    text, reas = "", ""
    for ch in s:
        for c in ch.choices:
            text += c.delta.content or ""
            reas += (c.delta.model_extra or {}).get("reasoning_content") or ""
    need(reas, "no reasoning deltas")
    need("<think>" not in text and "</think>" not in text, repr(text[:80]))
    need("no" in text.lower(), repr(text))
    return f"reasoning {len(reas)} chars, content {text.strip()[:40]!r}"


@test
def finish_length():
    r = chat([{"role": "user", "content": "Write a long story about a dragon."}], max_tokens=20, temperature=0, extra_body=NOTHINK)
    need(r.choices[0].finish_reason == "length", r.choices[0].finish_reason)
    need(r.usage.completion_tokens == 20, f"completion_tokens {r.usage.completion_tokens}")


@test
def stop_sequence():
    r = chat([{"role": "user", "content": "List the numbers from 1 to 20, one per line."}], max_tokens=200, temperature=0,
             stop=["7"], extra_body=NOTHINK)
    c = r.choices[0].message.content or ""
    need("7" not in c and "6" in c, repr(c))
    need(r.choices[0].finish_reason == "stop", r.choices[0].finish_reason)


@test
def multi_turn_system():
    msgs = [{"role": "system", "content": "You are a pirate. Always include the word 'arr' in your replies."},
            {"role": "user", "content": "My name is Ola. Say hi."}]
    r1 = chat(msgs, max_tokens=60, temperature=0, extra_body=NOTHINK)
    msgs += [{"role": "assistant", "content": r1.choices[0].message.content},
             {"role": "user", "content": "What is my name? One word."}]
    r2 = chat(msgs, max_tokens=60, temperature=0, extra_body=NOTHINK)
    c1, c2 = r1.choices[0].message.content, r2.choices[0].message.content
    need("arr" in c1.lower(), repr(c1))
    need("ola" in c2.lower(), repr(c2))
    return repr(c2)


@test
def unicode_polish():
    r = chat([{"role": "user", "content": "Przetłumacz na angielski: 'Zażółć gęślą jaźń'. Odpowiedz tylko tłumaczeniem. 🙂"}],
             max_tokens=60, temperature=0, extra_body=NOTHINK)
    c = r.choices[0].message.content or ""
    need(c.strip() and "�" not in c, repr(c))
    return repr(c[:60])


@test
def seed_reproducible():
    kw = dict(max_tokens=40, temperature=0.9, seed=1234, extra_body=NOTHINK)
    m = [{"role": "user", "content": "Invent a name for a fantasy city."}]
    a = chat(m, **kw).choices[0].message.content
    b = chat(m, **kw).choices[0].message.content
    need(a == b, f"{a!r} vs {b!r}")
    return repr(a[:40])


@test
def logprobs():
    r = chat([{"role": "user", "content": "Say hello."}], max_tokens=5, temperature=0, logprobs=True, top_logprobs=3, extra_body=NOTHINK)
    lp = r.choices[0].logprobs
    need(lp and lp.content and len(lp.content[0].top_logprobs) == 3, f"logprobs {lp}")
    return f"{lp.content[0].token!r} {lp.content[0].logprob:.3f}"


# ---------------------------------------------------------------- tools

def first_call(r):
    calls = r.choices[0].message.tool_calls or []
    need(calls, f"no tool_calls; content {r.choices[0].message.content!r}")
    return calls


@test
def tool_single():
    r = chat([{"role": "user", "content": "What's the weather in Krakow? Use the tool."}], tools=[WEATHER],
             max_tokens=256, temperature=0, extra_body=NOTHINK)
    c = first_call(r)[0]
    a = json.loads(c.function.arguments)
    need(c.function.name == "get_weather" and "krak" in a.get("city", "").lower(), f"{c.function.name} {a}")
    need(r.choices[0].finish_reason == "tool_calls", r.choices[0].finish_reason)
    need(c.id, "missing tool call id")
    return f"{c.function.name}({c.function.arguments})"


@test
def tool_single_thinking():
    r = chat([{"role": "user", "content": "What's the weather in Oslo in fahrenheit? Use the tool."}], tools=[WEATHER],
             max_tokens=2048, temperature=0)
    c = first_call(r)[0]
    a = json.loads(c.function.arguments)
    need("oslo" in a.get("city", "").lower() and a.get("unit") == "fahrenheit", str(a))
    return c.function.arguments


@test
def tool_parallel():
    r = chat([{"role": "user", "content": "Get the weather in Paris and in Tokyo. Call the tool for both cities at once."}],
             tools=[WEATHER], max_tokens=400, temperature=0, extra_body=NOTHINK)
    calls = first_call(r)
    cities = sorted(json.loads(c.function.arguments).get("city", "").lower() for c in calls)
    need(len(calls) == 2 and any("paris" in x for x in cities) and any("tokyo" in x for x in cities), str(cities))
    need(len({c.id for c in calls}) == 2, "duplicate tool call ids")
    return str(cities)


@test
def tool_complex_args():
    r = chat([{"role": "user", "content": "Book a table at 'Pod Aniołami' for 4 people, outdoor seating, we want pierogi and "
               "bigos. Contact: Jan Kowalski, phone +48 600 100 200."}],
             tools=[BOOK], max_tokens=400, temperature=0, extra_body=NOTHINK)
    a = json.loads(first_call(r)[0].function.arguments)
    need(isinstance(a.get("party_size"), int) and a["party_size"] == 4, f"party_size {a.get('party_size')!r}")
    need(a.get("outdoor") is True, f"outdoor {a.get('outdoor')!r}")
    need(isinstance(a.get("dishes"), list) and len(a["dishes"]) == 2, f"dishes {a.get('dishes')!r}")
    need(isinstance(a.get("contact"), dict) and "600" in a["contact"].get("phone", ""), f"contact {a.get('contact')!r}")
    need("Anio" in a.get("restaurant", ""), f"restaurant {a.get('restaurant')!r}")
    return json.dumps(a, ensure_ascii=False)[:120]


@test
def tool_round_trip():
    msgs = [{"role": "user", "content": "What's the weather in Krakow? Use the tool, then tell me in one sentence."}]
    r = chat(msgs, tools=[WEATHER], max_tokens=256, temperature=0, extra_body=NOTHINK)
    c = first_call(r)[0]
    msgs += [{"role": "assistant", "content": r.choices[0].message.content, "tool_calls": [c.model_dump()]},
             {"role": "tool", "tool_call_id": c.id, "content": json.dumps({"city": "Krakow", "temp_c": 17, "sky": "light rain"})}]
    r2 = chat(msgs, tools=[WEATHER], max_tokens=200, temperature=0, extra_body=NOTHINK)
    t = r2.choices[0].message.content or ""
    need("17" in t and "rain" in t.lower(), repr(t))
    need(not r2.choices[0].message.tool_calls, "called a tool again")
    return repr(t[:80])


@test
def tool_two_step_loop():
    # needs get_time for Tokyo and get_weather for Tokyo; executes until a final answer (thinking on, as agents run)
    msgs = [{"role": "user", "content": "I'm about to call a friend in Tokyo. What time is it there and what's the weather? "
             "Use the tools, then answer briefly."}]
    tools = [WEATHER, TIME]
    seen = set()
    for step in range(6):
        r = chat(msgs, tools=tools, max_tokens=3000, temperature=0)
        m = r.choices[0].message
        if not m.tool_calls:
            t = m.content or ""
            need({"get_time", "get_weather"} <= seen, f"tools used {seen}")
            need("14:05" in t or "2:05" in t, f"answer {t!r}")
            return f"{step} steps, tools {sorted(seen)}"
        msgs.append({"role": "assistant", "content": m.content, "tool_calls": [c.model_dump() for c in m.tool_calls]})
        for c in m.tool_calls:
            seen.add(c.function.name)
            res = {"time": "14:05"} if c.function.name == "get_time" else {"temp_c": 21, "sky": "clear"}
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res)})
    raise AssertionError("no final answer after 6 steps")


@test
def tool_choice_none():
    r = chat([{"role": "user", "content": "What's the weather in Krakow?"}], tools=[WEATHER], tool_choice="none",
             max_tokens=100, temperature=0, extra_body=NOTHINK)
    need(not r.choices[0].message.tool_calls, "tool called despite tool_choice none")
    need(r.choices[0].message.content, "empty content")


@test
def tool_choice_required():
    r = chat([{"role": "user", "content": "Hello there!"}], tools=[WEATHER], tool_choice="required",
             max_tokens=200, temperature=0, extra_body=NOTHINK)
    first_call(r)


@test
def tool_choice_named():
    r = chat([{"role": "user", "content": "I'm going to Rome tomorrow."}], tools=[WEATHER, TIME],
             tool_choice={"type": "function", "function": {"name": "get_time"}}, max_tokens=200, temperature=0, extra_body=NOTHINK)
    c = first_call(r)[0]
    need(c.function.name == "get_time", c.function.name)


@test
def tool_not_needed():
    r = chat([{"role": "user", "content": "What is 2 + 2? Answer with the number only."}], tools=[WEATHER],
             max_tokens=64, temperature=0, extra_body=NOTHINK)
    need(not r.choices[0].message.tool_calls, "tool called")
    need("4" in (r.choices[0].message.content or ""), repr(r.choices[0].message.content))


@test
def tool_stream():
    s = chat([{"role": "user", "content": "Get the weather in Paris and in Tokyo. Call the tool for both cities at once."}],
             tools=[WEATHER], max_tokens=400, temperature=0, stream=True, extra_body=NOTHINK)
    calls, fin = {}, None
    for ch in s:
        for c in ch.choices:
            fin = c.finish_reason or fin
            for tc in c.delta.tool_calls or []:
                d = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                d["id"] += tc.id or ""
                if tc.function:
                    d["name"] += tc.function.name or ""
                    d["args"] += tc.function.arguments or ""
    need(len(calls) == 2, f"{len(calls)} streamed calls")
    for d in calls.values():
        need(d["id"] and d["name"] == "get_weather", str(d))
        json.loads(d["args"])
    need(fin == "tool_calls", f"finish {fin}")
    return " ".join(d["args"] for d in calls.values())


@test
def tool_stream_thinking():
    s = chat([{"role": "user", "content": "Weather in Lisbon please, use the tool."}], tools=[WEATHER], max_tokens=2048,
             temperature=0, stream=True)
    args_, name, reas = "", "", ""
    for ch in s:
        for c in ch.choices:
            reas += (c.delta.model_extra or {}).get("reasoning_content") or ""
            for tc in c.delta.tool_calls or []:
                if tc.function:
                    name += tc.function.name or ""
                    args_ += tc.function.arguments or ""
    need(name == "get_weather" and "lisbon" in json.loads(args_).get("city", "").lower(), f"{name} {args_}")
    need(reas, "no reasoning")


WRITE = {"type": "function", "function": {
    "name": "write_file", "description": "Write a text file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                   "required": ["path", "content"]}}}


@test
def tool_code_argument():
    r = chat([{"role": "user", "content": "Use write_file to create hello.py containing a Python function greet(name) that "
               "returns f'<b>Hello, {name}!</b>' and a main block that prints greet(\"World\") and the string 'a && b < c'. Indent with 4 spaces."}], tools=[WRITE], max_tokens=600, temperature=0, extra_body=NOTHINK)
    c = first_call(r)[0]
    a = json.loads(c.function.arguments)
    code = a.get("content", "")
    need(a.get("path", "").endswith("hello.py"), str(a.get("path")))
    need("def greet(name)" in code and "<b>Hello" in code and "__main__" in code and "&&" in code, repr(code[:200]))
    compile(code, "hello.py", "exec")   # the file content must survive the XML tool format intact
    return f"{len(code)} chars of valid Python"


def agent_tools():
    names = ["read_file", "write_file", "edit_file", "list_dir", "grep", "glob", "run_shell", "git_status", "git_diff",
             "git_commit", "git_log", "web_fetch", "web_search", "todo_write", "ask_user", "run_tests", "lint", "format_code",
             "open_browser", "screenshot", "db_query", "http_request", "install_package", "create_dir"]
    tools = []
    for n in names:
        props = {"path": {"type": "string", "description": "Path relative to the repository root"}} if n in (
            "read_file", "list_dir", "create_dir", "lint", "format_code") else {"arg": {"type": "string"}}
        tools.append({"type": "function", "function": {"name": n, "description": f"Tool {n.replace('_', ' ')}.",
                      "parameters": {"type": "object", "properties": props, "required": list(props)}}})
    return tools


@test
def tool_many_tools():
    r = chat([{"role": "system", "content": "You are a coding agent working in a git repository."},
              {"role": "user", "content": "Show me what files are in the src directory."}],
             tools=agent_tools(), max_tokens=300, temperature=0, extra_body=NOTHINK)
    c = first_call(r)[0]
    need(c.function.name in ("list_dir", "glob", "run_shell"), c.function.name)
    return f"{c.function.name}({c.function.arguments})"


@test
def agent_session_cache():
    # a coding-agent session: big system prompt + 24 tools, five tool turns; each turn must only process new tokens
    sysmsg = "You are a coding agent. Repository notes:\n" + DOC
    msgs = [{"role": "system", "content": sysmsg},
            {"role": "user", "content": "Read the files src/a.py, src/b.py, src/c.py, src/d.py and src/e.py one at a time "
             "(one read_file call per turn), then summarise them in one sentence."}]
    news, t0 = [], time.time()
    for step in range(8):
        r = chat(msgs, tools=agent_tools(), max_tokens=1500, temperature=0)
        news.append(timings(r).get("prompt_n"))
        m = r.choices[0].message
        if not m.tool_calls:
            break
        msgs.append({"role": "assistant", "content": m.content, "tool_calls": [c.model_dump() for c in m.tool_calls]})
        for c in m.tool_calls:
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": f"# file {json.loads(c.function.arguments)}\nx = {step}\n"})
    need(len(news) >= 3, f"only {len(news)} turns")
    need(max(news[1:]) < 600, f"new prompt tokens per turn {news}")
    return f"{len(news)} turns, new prompt tokens {news}, {time.time() - t0:.0f}s"


# ---------------------------------------------------------------- structured output

@test
def json_object():
    r = chat([{"role": "user", "content": "Give me a JSON object with keys 'name' (string) and 'age' (number) for a "
               "fictional person."}], response_format={"type": "json_object"}, max_tokens=100, temperature=0, extra_body=NOTHINK)
    o = json.loads(r.choices[0].message.content)
    need(isinstance(o, dict) and "name" in o, str(o))
    return json.dumps(o)


@test
def json_schema():
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"},
              "landmarks": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3}},
              "required": ["city", "population", "landmarks"], "additionalProperties": False}
    r = chat([{"role": "user", "content": "Describe Krakow."}], max_tokens=300, temperature=0, extra_body=NOTHINK,
             response_format={"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": schema}})
    o = json.loads(r.choices[0].message.content)
    need(set(o) == {"city", "population", "landmarks"}, str(o))
    need(isinstance(o["population"], int) and 2 <= len(o["landmarks"]) <= 3, str(o))
    return json.dumps(o, ensure_ascii=False)[:100]


# ---------------------------------------------------------------- prompt caching





@test
def cache_prefix():
    base = [{"role": "system", "content": "You answer questions about the document.\n\n" + DOC}]
    r1 = chat(base + [{"role": "user", "content": "What is the value of item 12?"}], max_tokens=16, temperature=0, extra_body=NOTHINK)
    r2 = chat(base + [{"role": "user", "content": "What is the value of item 345?"}], max_tokens=16, temperature=0, extra_body=NOTHINK)
    p1, p2 = timings(r1).get("prompt_n"), timings(r2).get("prompt_n")
    need(p2 < 64, f"prompt_n {p1} -> {p2}")
    return f"prompt_n {p1} -> {p2}"


@test
def cache_multi_turn_thinking():
    # agents keep extending one conversation; each turn must only process the new tokens, also with thinking on
    msgs = [{"role": "system", "content": "You answer questions about the document.\n\n" + DOC},
            {"role": "user", "content": "What is the value of item 12? Answer briefly."}]
    r1 = chat(msgs, max_tokens=1500, temperature=0)
    msgs += [{"role": "assistant", "content": r1.choices[0].message.content},
             {"role": "user", "content": "And item 13? Answer briefly."}]
    r2 = chat(msgs, max_tokens=1500, temperature=0)
    t1, t2 = timings(r1), timings(r2)
    need(t2.get("prompt_n", 1e9) < 400, f"turn 2 reprocessed {t2.get('prompt_n')} tokens (turn 1: {t1.get('prompt_n')})")
    return f"turn1 {t1.get('prompt_n')} -> turn2 {t2.get('prompt_n')} new tokens"


@test
def cache_tool_loop():
    msgs = [{"role": "system", "content": "You are a helpful assistant.\n\n" + DOC},
            {"role": "user", "content": "What's the weather in Krakow? Use the tool."}]
    r1 = chat(msgs, tools=[WEATHER], max_tokens=1500, temperature=0)
    c = first_call(r1)[0]
    msgs += [{"role": "assistant", "content": r1.choices[0].message.content, "tool_calls": [c.model_dump()]},
             {"role": "tool", "tool_call_id": c.id, "content": json.dumps({"temp_c": 17})}]
    r2 = chat(msgs, tools=[WEATHER], max_tokens=1500, temperature=0)
    p2 = timings(r2).get("prompt_n", 1e9)
    need(p2 < 400, f"tool turn reprocessed {p2} tokens")
    return f"after tool result: {p2} new tokens"


# ---------------------------------------------------------------- other endpoints

@test
def completions_raw():
    r = oc.completions.create(model=MODEL, prompt="The capital of Poland is", max_tokens=8, temperature=0)
    need("warsaw" in r.choices[0].text.lower(), repr(r.choices[0].text))


@test
def anthropic_basic():
    r = ac.messages.create(model=MODEL, max_tokens=64, messages=[{"role": "user", "content": "Reply with one word: the capital of Italy."}],
                           extra_body=NOTHINK)
    t = "".join(b.text for b in r.content if b.type == "text")
    need("rome" in t.lower(), repr(t))
    need(r.stop_reason == "end_turn", r.stop_reason)
    return repr(t)


@test
def anthropic_thinking():
    r = ac.messages.create(model=MODEL, max_tokens=2048, messages=[{"role": "user", "content": "What is 12 * 12? Number only."}],
                           thinking={"type": "enabled", "budget_tokens": 1024})
    kinds = [b.type for b in r.content]
    t = "".join(b.text for b in r.content if b.type == "text")
    need("144" in t, repr(t))
    need("thinking" in kinds, f"blocks {kinds}")
    return str(kinds)


@test
def anthropic_claude_code_style():
    # system as cache_control blocks, tools, thinking, streaming - the shape Claude-Code-like clients send
    tools = [{"name": "Read", "description": "Read a file.", "input_schema": {"type": "object", "properties": {
                "file_path": {"type": "string"}}, "required": ["file_path"]}},
             {"name": "Bash", "description": "Run a shell command.", "input_schema": {"type": "object", "properties": {
                "command": {"type": "string"}}, "required": ["command"]}}]
    system = [{"type": "text", "text": "You are an interactive CLI coding assistant.", "cache_control": {"type": "ephemeral"}},
              {"type": "text", "text": "Working directory: /home/user/project"}]
    with ac.messages.stream(model=MODEL, max_tokens=2048, system=system, tools=tools,
                            thinking={"type": "enabled", "budget_tokens": 1024},
                            messages=[{"role": "user", "content": [{"type": "text", "text": "What does README.md say? Read it."}]}]) as s:
        final = s.get_final_message()
    tu = [b for b in final.content if b.type == "tool_use"]
    need(tu and tu[0].name == "Read" and "readme" in tu[0].input.get("file_path", "").lower(), str(final.content))
    need(final.stop_reason == "tool_use", final.stop_reason)
    return str(tu[0].input)


@test
def anthropic_tools():
    tools = [{"name": "get_weather", "description": "Get the current weather for a city.",
              "input_schema": WEATHER["function"]["parameters"]}]
    msgs = [{"role": "user", "content": "What's the weather in Krakow? Use the tool."}]
    r = ac.messages.create(model=MODEL, max_tokens=512, tools=tools, messages=msgs, extra_body=NOTHINK)
    tu = [b for b in r.content if b.type == "tool_use"]
    need(tu and tu[0].name == "get_weather" and "krak" in tu[0].input.get("city", "").lower(), str(r.content))
    need(r.stop_reason == "tool_use", r.stop_reason)
    msgs += [{"role": "assistant", "content": r.content},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu[0].id, "content": "17 C, light rain"}]}]
    r2 = ac.messages.create(model=MODEL, max_tokens=200, tools=tools, messages=msgs, extra_body=NOTHINK)
    t = "".join(b.text for b in r2.content if b.type == "text")
    need("17" in t, repr(t))
    return repr(t[:60])


@test
def anthropic_stream_tools():
    tools = [{"name": "get_weather", "description": "Get the current weather for a city.",
              "input_schema": WEATHER["function"]["parameters"]}]
    with ac.messages.stream(model=MODEL, max_tokens=512, tools=tools, extra_body=NOTHINK,
                            messages=[{"role": "user", "content": "Weather in Berlin? Use the tool."}]) as s:
        final = s.get_final_message()
    tu = [b for b in final.content if b.type == "tool_use"]
    need(tu and "berlin" in tu[0].input.get("city", "").lower(), str(final.content))


@test
def anthropic_tool_choice():
    tools = [{"name": n["function"]["name"], "description": n["function"]["description"],
              "input_schema": n["function"]["parameters"]} for n in (WEATHER, TIME)]
    r = ac.messages.create(model=MODEL, max_tokens=300, tools=tools, tool_choice={"type": "tool", "name": "get_time"},
                           messages=[{"role": "user", "content": "I'm going to Rome tomorrow."}], extra_body=NOTHINK)
    tu = [b for b in r.content if b.type == "tool_use"]
    need(tu and tu[0].name == "get_time", str(r.content))
    r = ac.messages.create(model=MODEL, max_tokens=300, tools=tools, tool_choice={"type": "any"},
                           messages=[{"role": "user", "content": "Hi!"}], extra_body=NOTHINK)
    need(any(b.type == "tool_use" for b in r.content), str(r.content))


@test
def responses_tool_choice():
    tools = [{"type": "function", "name": n["function"]["name"], "description": n["function"]["description"],
              "parameters": n["function"]["parameters"]} for n in (WEATHER, TIME)]
    r = oc.responses.create(model=MODEL, input="I'm going to Rome tomorrow.", tools=tools, max_output_tokens=300,
                            tool_choice={"type": "function", "name": "get_weather"}, extra_body=NOTHINK)
    fc = [o for o in r.output if o.type == "function_call"]
    need(fc and fc[0].name == "get_weather", str(r.output))


@test
def tool_choice_required_thinking():
    r = chat([{"role": "user", "content": "Hello there!"}], tools=[WEATHER], tool_choice="required", max_tokens=2048, temperature=0)
    first_call(r)


@test
def responses_basic():
    r = oc.responses.create(model=MODEL, input="Reply with one word: the capital of Spain.", max_output_tokens=64,
                            extra_body=NOTHINK)
    need("madrid" in (r.output_text or "").lower(), repr(r.output_text))
    return repr(r.output_text)


@test
def responses_tools():
    tools = [{"type": "function", "name": "get_weather", "description": "Get the current weather for a city.",
              "parameters": WEATHER["function"]["parameters"]}]
    r = oc.responses.create(model=MODEL, input="What's the weather in Krakow? Use the tool.", tools=tools,
                            max_output_tokens=512, extra_body=NOTHINK)
    fc = [o for o in r.output if o.type == "function_call"]
    need(fc and fc[0].name == "get_weather" and "krak" in json.loads(fc[0].arguments).get("city", "").lower(), str(r.output))
    r2 = oc.responses.create(model=MODEL, tools=tools, max_output_tokens=200, extra_body=NOTHINK,
                             input=[{"role": "user", "content": "What's the weather in Krakow? Use the tool."}, fc[0].model_dump(),
                                    {"type": "function_call_output", "call_id": fc[0].call_id, "output": "17 C, light rain"}])
    need("17" in (r2.output_text or ""), repr(r2.output_text))
    return repr(r2.output_text[:60])


# ---------------------------------------------------------------- robustness

@test
def disconnect_mid_stream():
    s = chat([{"role": "user", "content": "Write a 2000-word essay about lighthouses."}], max_tokens=3000, temperature=0,
             stream=True, extra_body=NOTHINK)
    n = 0
    for _ in s:
        n += 1
        if n >= 20:
            break
    s.close()
    t0 = time.time()
    r = chat([{"role": "user", "content": "Say OK."}], max_tokens=5, temperature=0, extra_body=NOTHINK)
    dt = time.time() - t0
    need(dt < 10, f"next request waited {dt:.1f}s (generation not cancelled?)")
    return f"next request {dt:.1f}s"


@test
def concurrent_requests():
    out = [None, None]

    def run(i):
        out[i] = chat([{"role": "user", "content": f"Reply with the number {i + 5} only."}], max_tokens=10, temperature=0,
                      extra_body=NOTHINK).choices[0].message.content
    th = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    [x.start() for x in th]
    [x.join() for x in th]
    need(out[0] and "5" in out[0] and out[1] and "6" in out[1], str(out))


@test
def long_output_coherent():
    r = chat([{"role": "user", "content": "Write a detailed 1500-word technical article about how lighthouses work, "
               "with section headings."}], max_tokens=2500, temperature=0.7, top_p=0.95, seed=7, extra_body=NOTHINK)
    c = r.choices[0].message.content or ""
    words = c.split()
    need(len(words) > 600, f"only {len(words)} words")
    # degenerate output: long runs of one token or a tiny vocabulary
    uniq = len(set(words)) / max(1, len(words))
    need(uniq > 0.2, f"unique-word ratio {uniq:.2f}")
    need("�" not in c, "replacement characters")
    tl = timings(r)
    return f"{len(words)} words, uniq {uniq:.2f}, {tl.get('predicted_per_second', 0):.1f} t/s"


if "--long" in sys.argv:
    @test
    def long_context_needle():
        # ~100k tokens: above the sparse-attention threshold (96k); the answer sits near the start
        filler = open("/home/vene/jt/docs_all.txt", errors="ignore").read()
        body = ("The secret code word for this conversation is 'aubergine-7'.\n\n" + filler)[:400000]
        t0 = time.time()
        r = chat([{"role": "user", "content": body + "\n\nWhat is the secret code word mentioned at the very start? Answer with the word only."}],
                 max_tokens=40, temperature=0, extra_body=NOTHINK)
        c = r.choices[0].message.content or ""
        need("aubergine-7" in c.lower(), repr(c))
        tl = timings(r)
        return f"{tl.get('prompt_n')} prompt tokens, prefill {tl.get('prompt_per_second', 0):.0f} t/s, {time.time() - t0:.0f}s, {c!r}"


print()
fails = [r for r in results if not r[1]]
print(f"{len(results) - len(fails)}/{len(results)} passed")
for n, _, d in fails:
    print(f"  FAIL {n}: {d[:300]}")
sys.exit(1 if fails else 0)

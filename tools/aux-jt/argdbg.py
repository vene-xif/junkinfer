import json, urllib.request
WRITE = {"type": "function", "function": {"name": "write_file", "description": "Write a text file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}
body = {"messages": [{"role": "user", "content": "Use write_file to create hello.py containing a Python function greet(name) that returns f'<b>Hello, {name}!</b>' and a main block that prints greet(\"World\") and the string 'C:\\\\temp\\\\x && y'. Indent with 4 spaces."}],
        "tools": [WRITE], "max_tokens": 600, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req))
args = d["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
print("RAW ARGUMENTS JSON:", args)
print("CONTENT:"); print(json.loads(args)["content"])

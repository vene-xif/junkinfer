import json, urllib.request, sys
U="http://127.0.0.1:8080/v1/chat/completions"
tools=[{"type":"function","function":{"name":"read_file","description":"Read a file from disk","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
       {"type":"function","function":{"name":"terminal","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
msgs=[{"role":"system","content":"You are an agent. Use tools when needed."},{"role":"user","content":"List the files in /tmp using the terminal tool."}]
def call(stream):
    body={"messages":msgs,"tools":tools,"max_tokens":400,"temperature":0.6,"stream":stream}
    r=urllib.request.urlopen(urllib.request.Request(U,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600)
    if not stream: return json.load(r)["choices"][0]
    content="";tcs={};fr=None;reason=""
    for line in r:
        line=line.decode().strip()
        if not line.startswith("data: ") or line=="data: [DONE]": continue
        d=json.loads(line[6:]);c=d["choices"][0] if d.get("choices") else None
        if not c: continue
        dl=c.get("delta",{});content+=dl.get("content") or "";reason+=dl.get("reasoning_content") or ""
        for tc in dl.get("tool_calls") or []:
            t=tcs.setdefault(tc["index"],{"name":"","args":""})
            f=tc.get("function",{});t["name"]+=f.get("name") or "";t["args"]+=f.get("arguments") or ""
        fr=c.get("finish_reason") or fr
    return {"content":content,"tool_calls":tcs,"finish_reason":fr,"reasoning_len":len(reason)}
for s in (False,True):
    r=call(s); print("stream" if s else "plain", json.dumps(r)[:400])
# turn 2 with a tool result
msgs.append({"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"terminal","arguments":json.dumps({"command":"ls /tmp"})}}]})
msgs.append({"role":"tool","tool_call_id":"c1","content":"a.txt\nb.log\n"})
msgs.append({"role":"user","content":"Now read a.txt."})
r=call(True); print("turn2", json.dumps(r)[:400])

#!/bin/bash
# fresh server with the given env, one completion on p8k
~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache "$@" > /dev/null
python3 - <<'PY'
import json, urllib.request, os
p = open(os.path.expanduser("~/jt/p8k.txt")).read()
d = json.dumps({"prompt": p, "n_predict": 48, "temperature": 0, "cache_prompt": True}).encode()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/completion", d, {"Content-Type": "application/json"})))
print(repr(r["content"][100:160]))
PY

#!/bin/bash
~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache "$@" > /dev/null
python3 - <<'PY'
import json, urllib.request, os
p = open(os.path.expanduser("~/jt/p8k.txt")).read()
d = json.dumps({"prompt": p, "n_predict": 1, "temperature": 0, "n_probs": 4, "cache_prompt": False}).encode()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/completion", d, {"Content-Type": "application/json"})))
print([round(t["logprob"], 5) for t in r["completion_probabilities"][0]["top_logprobs"]])
PY

#!/bin/bash
# save with env A, restore with env B: srcross.sh "A" "B"
U=http://127.0.0.1:8080
req() { python3 - "$1" <<'PY'
import json, sys, urllib.request
p = open(sys.argv[1]).read()
d = json.dumps({"prompt": p, "n_predict": 48, "temperature": 0, "cache_prompt": True}).encode()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/completion", d, {"Content-Type": "application/json"})))
print(repr(r["content"][:160]), r["timings"]["prompt_n"])
PY
}
~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache $1 > /dev/null
req ~/jt/p8k.txt
curl -s -X POST "$U/slots/0?action=save" -H "Content-Type: application/json" -d "{\"filename\":\"srtest.bin\"}" > /dev/null
~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache $2 > /dev/null
curl -s -X POST "$U/slots/0?action=restore" -H "Content-Type: application/json" -d "{\"filename\":\"srtest.bin\"}" | cut -c1-60; echo
req ~/jt/p8k.txt

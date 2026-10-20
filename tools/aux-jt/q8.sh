#!/bin/bash
cd ~/junkinfer/engine && cmake --build build -j12 2>&1 | grep -E "error" | head
cd ~ && ROW=1 ~/jt/spfgate.sh "JI_SPF_TAIL_LAYERS=7,11" "JI_SPF_TAIL_LAYERS=3,7,11" "JI_SPF_TAIL_LAYERS=7,11,15,19" 2>&1 | sed 's/JR_STATE_DIR=[^ ]* //' | cut -c1-110
for e in JI_SPF_TAIL_BUDGET=16384 JI_SPF_TAIL_LAYERS=7,11 JI_SPF_TAIL_LAYERS=3,7,11; do ~/jt/restart.sh $e >/dev/null; sleep 20; echo -n "$e: "; ~/jt/venv/bin/python ~/jt/srvpf.py ~/jt/docs_all.txt 2>&1 | tail -1; done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}"); kill $P

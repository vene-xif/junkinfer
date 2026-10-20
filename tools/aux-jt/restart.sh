#!/bin/bash
# guard: never two model processes at once (pinned experts OOM the box)
O=$(ps -eo pid,args | grep -E "[b]in/llama-(junk-ref|completion|perplexity|bench|cli)" | awk "{print \$1}")
if [ -n "$O" ]; then echo "refusing: another model process is running ($O)"; exit 1; fi
# restart the EXL3 server with the given env assignments (e.g. JI_ELASTIC=0); kills the running one by PID
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}")
[ -n "$P" ] && kill $P && sleep 5
sync; echo kot | sudo -S -p "" sh -c "echo 3 > /proc/sys/vm/drop_caches"   # page cache full = slower 245k decode (2026-10-20)
cd ~/junkinfer
(env "$@" nohup scripts/serve.sh ${MODEL:-/mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf} -v > ~/jt/srv.log 2>&1 &)
for i in $(seq 1 80); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 5; done
echo "server up"

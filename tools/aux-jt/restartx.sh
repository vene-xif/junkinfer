#!/bin/bash
# guard: never two model processes at once (pinned experts OOM the box)
O=$(ps -eo pid,args | grep -E "[b]in/llama-(junk-ref|completion|perplexity|bench|cli)" | awk "{print \$1}")
if [ -n "$O" ]; then echo "refusing: another model process is running ($O)"; exit 1; fi
# restart the server with extra llama-server args: restartx.sh "ENV=.. ENV2=.." args...
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}")
[ -n "$P" ] && kill $P && sleep 5
ENVS=$1; shift
cd ~/junkinfer
(env $ENVS nohup scripts/serve.sh /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -v "$@" > ~/jt/srv.log 2>&1 &)
for i in $(seq 1 80); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 5; done
echo "server up"

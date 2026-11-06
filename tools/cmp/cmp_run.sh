#!/bin/bash
# Engine comparison runner (AUX1). Usage: cmp_run.sh label [label...]; labels: stock1 stock2 ji1 ji2nomtp ji2
# Each config: start server (PID saved), wait for /health, run cmp_bench.py, stop the server (idle) by PID.
cd ~/cmp
C=~/tinyinfer/data/corpus_full.txt
CT=$HOME/junkinfer/ref/templates/qwen_sharp.jinja
STOCK=~/llama.cpp-master/build/bin/llama-server
IQ4=~/models/ssd-overflow/Qwen3.6-35B-IQ4_NL.gguf
EXL3=/mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf
for L in "$@"; do
  echo "=== $L start $(date -Is)" >> runs.log
  case $L in
    stock1) GGML_VK_VISIBLE_DEVICES=1 RADV_PERFTEST=nogttspill setsid nohup $STOCK -m $IQ4 --fit on -fa on -c 40960 -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -t 6 -np 1 --jinja --chat-template-file $CT --host 127.0.0.1 --port 8080 > srv_$L.log 2>&1 < /dev/null & ;;
    stock2) RADV_PERFTEST=nogttspill setsid nohup $STOCK -m $IQ4 --fit on -fa on -c 40960 -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -t 6 -np 1 --jinja --chat-template-file $CT --host 127.0.0.1 --port 8080 > srv_$L.log 2>&1 < /dev/null & ;;
    ji1) JI_TP=0 JI_CTX=40960 JI_HOST=127.0.0.1 JI_MODEL=$EXL3 setsid nohup ~/junkinfer/scripts/serve_boot.sh > srv_$L.log 2>&1 < /dev/null & ;;
    ji2nomtp) JI_TP=2 JI_MTP=0 JI_CTX=40960 JI_HOST=127.0.0.1 JI_MODEL=$EXL3 setsid nohup ~/junkinfer/scripts/serve_boot.sh > srv_$L.log 2>&1 < /dev/null & ;;
    ji2) JI_TP=2 JI_MTP=1 JI_CTX=40960 JI_HOST=127.0.0.1 JI_MODEL=$EXL3 setsid nohup ~/junkinfer/scripts/serve_boot.sh > srv_$L.log 2>&1 < /dev/null & ;;
  esac
  P=$!; echo $P > server.pid
  ok=0
  for i in $(seq 1 240); do curl -sf http://127.0.0.1:8080/health > /dev/null && { ok=1; break; }; kill -0 $P 2>/dev/null || break; sleep 5; done
  if [ $ok = 1 ]; then
    for d in 0 1; do cat /sys/class/drm/card$d/device/mem_info_vram_used 2>/dev/null; done | tr '\n' ' ' >> runs.log; echo " (VRAM used after load, bytes)" >> runs.log
    python3 ~/cmp/cmp_bench.py $L $C >> runs.log 2>&1
  else
    echo "$L: server did not come up" >> runs.log
  fi
  kill $P 2>/dev/null; for i in $(seq 1 90); do kill -0 $P 2>/dev/null || break; sleep 1; done
  kill -0 $P 2>/dev/null && echo "$L: server still alive after SIGTERM+90s (left running, PID $P)" >> runs.log && break
  echo "=== $L done $(date -Is)" >> runs.log
  sleep 5
done
echo "CMP_ALL_DONE $(date -Is)" >> runs.log

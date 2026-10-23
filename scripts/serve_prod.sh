#!/usr/bin/env bash
# Production server on AUX-1 (two RX 580s): tensor-parallel decode (JI_TP=2), two-card JDE prefill (JI_JPF=1) with
# its helper scratch margin, slot save / restore to the SSD, listening on the LAN. Restarts a running server (by PID),
# runs detached, waits until it answers. Any VAR=value arguments override the defaults below, e.g.
#   scripts/serve_prod.sh                        # LAN, no API key
#   scripts/serve_prod.sh JI_API_KEY=secret      # with a key
#   scripts/serve_prod.sh JI_HOST=127.0.0.1      # local only
# Log: ~/junkinfer/logs/server.log (crashes: logs/restarts.log, logs/server.*.crash.log). Auto-restarts on exit.
# Stop: kill $(cat logs/supervisor.pid), then the server PID it prints.
set -euo pipefail
cd "$(dirname "$0")/.."

# never two model processes at once (pinned experts: the box runs out of memory)
other=$(ps -eo pid,args | grep -E "[b]in/llama-(junk-ref|completion|perplexity|bench|cli)" | awk '{print $1}' || true)
if [ -n "$other" ]; then echo "refusing: another model process is running ($other)"; exit 1; fi
# the supervisor first (it would restart the server), then the server
if [ -f logs/supervisor.pid ] && kill -0 "$(cat logs/supervisor.pid)" 2>/dev/null; then
  echo "stopping the supervisor ($(cat logs/supervisor.pid))"; kill "$(cat logs/supervisor.pid)"; sleep 1
fi
old=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk '{print $1}' || true)
if [ -n "$old" ]; then echo "stopping the running server ($old)"; kill $old; sleep 5; fi

# CPU governor performance (decode +6-9 %, see scripts/cpu_performance.sh): applied when sudo needs no password
if [ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" != performance ]; then
  sudo -n sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done' 2>/dev/null \
    || echo "note: CPU governor is not 'performance' (decode -6-9 %): run scripts/cpu_performance.sh install"
fi
export JI_HOST=${JI_HOST:-0.0.0.0} JI_PORT=${JI_PORT:-8080}
# MTP speculative decoding on by default (JI_MTP=0: off; decode +14-23 %, prefill -2-6 %); its head needs a larger
# helper margin
export JI_MTP=${JI_MTP:-1}
if [ "$JI_MTP" != 0 ]; then DEF_MARGIN=1900; else DEF_MARGIN=1500; fi
export JI_TP=2 JI_JPF=1 JI_TP_MARGIN_MIB=${JI_TP_MARGIN_MIB:-$DEF_MARGIN}
export JI_SLOT_DIR=${JI_SLOT_DIR:-/mnt/ssd/kvcache}
MODEL=${MODEL:-/mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf}
mkdir -p logs
# supervisor: restarts the server if it exits (crash, GPU hang abort); the crashed run's log is kept as
# logs/server.<time>.crash.log, restarts are listed in logs/restarts.log. Stop: kill $(cat logs/supervisor.pid), then the server.
env "$@" setsid nohup bash -c '
  while true; do
    scripts/serve.sh "$0" > logs/server.log 2>&1 < /dev/null
    rc=$?; t=$(date +%Y%m%d-%H%M%S)
    cp logs/server.log "logs/server.$t.crash.log"
    echo "$t server exited ($rc), restarting" >> logs/restarts.log
    sleep 5
  done' "$MODEL" > /dev/null 2>&1 < /dev/null &
echo $! > logs/supervisor.pid
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$JI_PORT/health" > /dev/null; then
    pid=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk '{print $1}')
    echo "server up (pid $pid): http://$(hostname -I | awk '{print $1}'):$JI_PORT/v1 (web UI at /)"
    exit 0
  fi
  sleep 5
done
echo "the server did not come up in 10 min; see logs/server.log"; exit 1

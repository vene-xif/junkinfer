#!/bin/bash
# GPU thermal logger / guard (systemd: scripts/ji-thermal-guard.service). Every 2 s: both GPUs' temperature, fan,
# power -> logs/thermal.log (the previous boot's log is kept as logs/thermal.prev.log); at >= LIMIT C (88) it sends
# SIGTERM to the llama process (JDE releases its TP events on SIGTERM; the server service restarts it).
LIMIT=${LIMIT:-88}
L=$(cd "$(dirname "$0")/.." && pwd)/logs
mkdir -p "$L"
[ -f "$L/thermal.log" ] && mv -f "$L/thermal.log" "$L/thermal.prev.log"
while true; do
  line="$(date '+%F %T')"; hot=0
  for c in /sys/class/drm/card[0-9]; do h=$(ls -d $c/device/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
    t=$(( $(cat $h/temp1_input)/1000 )); p=$(( $(cat $h/power1_average 2>/dev/null || echo 0)/1000000 ))
    line="$line $(basename $c):${t}C/$(cat $h/fan1_input)rpm/${p}W"; [ $t -ge $LIMIT ] && hot=1; done
  echo "$line" >> "$L/thermal.log"
  if [ $hot = 1 ]; then echo "$(date '+%F %T') LIMIT $LIMIT reached - stopping" >> "$L/thermal.log"
    P=$(ps -eo pid,args | grep -E "[b]in/llama-(server|junk-ref)" | awk '{print $1}'); [ -n "$P" ] && kill -TERM $P; fi
  sleep 2
done

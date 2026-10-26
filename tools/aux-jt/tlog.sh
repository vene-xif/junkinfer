#!/bin/bash
# tlog.sh: log both GPUs temp / fan / power every 2 s to ~/jt/thermal.log; SIGTERM the llama process if any GPU >= LIMIT C
LIMIT=${LIMIT:-88}
: > ~/jt/thermal.log
while true; do
  line="$(date +%T)"; hot=0
  for c in /sys/class/drm/card[0-9]; do d=$c/device; h=$(ls -d $d/hwmon/hwmon* | head -1)
    t=$(( $(cat $h/temp1_input)/1000 )); p=$(( $(cat $h/power1_average 2>/dev/null || echo 0)/1000000 ))
    line="$line $(basename $c):${t}C/$(cat $h/fan1_input)rpm/${p}W"; [ $t -ge $LIMIT ] && hot=1; done
  echo "$line" >> ~/jt/thermal.log
  if [ $hot = 1 ]; then echo "$(date +%T) LIMIT $LIMIT reached - stopping" >> ~/jt/thermal.log
    P=$(ps -eo pid,args | grep -E "[b]in/llama-(server|junk-ref)" | awk "{print \$1}"); [ -n "$P" ] && kill -TERM $P; fi
  sleep 2
done

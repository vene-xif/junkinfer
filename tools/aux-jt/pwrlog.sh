#!/bin/bash
# 1 Hz GPU log to the HDD: time, per card (pci) power W / temp C / sclk / gpu busy, plus the x4 link status
out=${1:-$HOME/jt/pwr.log}
while true; do
  l="$(date +%T)"
  for c in /sys/class/drm/card[0-9]; do
    d=$c/device; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1)
    l="$l | $(basename $(readlink -f $d)) $(( $(cat $h/power1_input 2>/dev/null || echo 0) / 1000000 ))W $(( $(cat $h/temp1_input 2>/dev/null || echo 0) / 1000 ))C $(grep '\*' $d/pp_dpm_sclk 2>/dev/null | awk '{print $2}') $(cat $d/gpu_busy_percent 2>/dev/null)%"
  done
  echo "$l" >> "$out"; sync "$out" 2>/dev/null
  sleep 1
done

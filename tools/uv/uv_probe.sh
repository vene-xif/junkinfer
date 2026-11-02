#!/bin/bash
# uv_probe.sh <card> <vkdev> <T>: samples active sclk state + vddgfx + power every 0.2 s during a 20 s stress
c=$1; dev=$2; T=$3; d=/sys/class/drm/card$c/device; h=$(ls -d $d/hwmon/hwmon* | head -1)
cd ~/junkinfer/lab/mmbench
MB_DEV=$dev MB_T=$T MB_STRESS=20 ./mmbench prod_f.spv:64,prod_h.spv:32 > /tmp/probe.out 2>&1 &
p=$!
sleep 4
for i in $(seq 1 40); do
  echo "$(grep '\*' $d/pp_dpm_sclk | tr -d '\n') $(cat $h/in0_input)mV $(( $(cat $h/power1_input)/1000000 ))W"
  sleep 0.2
done | sort | uniq -c | sort -rn | head -12
wait $p; grep -a "STRESS" /tmp/probe.out

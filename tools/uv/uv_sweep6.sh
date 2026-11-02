#!/bin/bash
# uv_sweep6.sh <card> <vkdev> <Vhi> <Vlo first> <Vlo last> <step>: like uv_sweep5 but sweeps Vlo (sclk states 2-3 and
# the m 2 floor) with the top states fixed at <Vhi>; heavy run first (it lives in states 2-3), then the light runs
c=$1; dev=$2; H=$3; v0=$4; v1=$5; st=$6
for L in $(seq $v0 -$st $v1); do
  ~/jt/uv/uv_set5.sh $c $L $H || { echo "SET FAILED $L"; exit 1; }
  echo "=== card$c Vlo $L Vhi $H: $(sed -n 4,13p /sys/class/drm/card$c/device/pp_od_clk_voltage | tr -s " " | tr "\n" " ")"
  e0=$(sudo -n dmesg | grep -ci "amdgpu.*\(error\|fault\|timeout\|hang\|reset\)")
  ok=1
  for T in 4000 150; do
    s=40; [ $T = 4000 ] && s=90
    r=$(MB_T_OVR=$T ~/jt/uv/stress1.sh $dev $s); echo "T=$T: $r"
    echo "$r" | grep -q "exit 0" || { ok=0; break; }
  done
  e1=$(sudo -n dmesg | grep -ci "amdgpu.*\(error\|fault\|timeout\|hang\|reset\)")
  if [ $ok = 1 ] && [ "$e1" = "$e0" ]; then echo "PASS $L"; else echo "FAIL at $L (dmesg new $((e1-e0)))"; exit 3; fi
done
echo SWEEP_END

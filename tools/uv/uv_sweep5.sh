#!/bin/bash
# uv_sweep5.sh <card> <vkdev> <Vlo> <Vhi first> <Vhi last> <step>: uv_set5 at each Vhi, then bit-exact stress:
# two light runs (T=64, T=150: top clock) and one heavy run (T=4000: power cap); stop at the first failure
c=$1; dev=$2; L=$3; v0=$4; v1=$5; st=$6
for H in $(seq $v0 -$st $v1); do
  ~/jt/uv/uv_set5.sh $c $L $H || { echo "SET FAILED $H"; exit 1; }
  echo "=== card$c Vlo $L Vhi $H: $(sed -n 4,9p /sys/class/drm/card$c/device/pp_od_clk_voltage | tr -s " " | tr "\n" " ")"
  e0=$(sudo -n dmesg | grep -ci "amdgpu.*\(error\|fault\|timeout\|hang\|reset\)")
  ok=1
  for T in 64 150 4000; do
    s=40; [ $T = 4000 ] && s=60
    r=$(MB_T_OVR=$T ~/jt/uv/stress1.sh $dev $s); echo "T=$T: $r"
    echo "$r" | grep -q "exit 0" || { ok=0; break; }
  done
  e1=$(sudo -n dmesg | grep -ci "amdgpu.*\(error\|fault\|timeout\|hang\|reset\)")
  if [ $ok = 1 ] && [ "$e1" = "$e0" ]; then echo "PASS $H"; else echo "FAIL at $H (dmesg new $((e1-e0)))"; exit 3; fi
done
echo SWEEP_END

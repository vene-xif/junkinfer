#!/bin/bash
# uv_sweep4.sh <card> <vkdev> <max MHz> <V first> <V last> <step> <sec heavy>: uv_set4 at each voltage, then a bit-exact
# heavy (power-capped) and light (top clock) stress; stops at the first mismatch / error exit / new amdgpu dmesg line
c=$1; dev=$2; F=$3; v0=$4; v1=$5; st=$6; sec=$7
while ! pgrep -x mmbench >/dev/null; do break; done
while pgrep -x mmbench >/dev/null; do sleep 2; done
for V in $(seq $v0 -$st $v1); do
  ~/jt/uv/uv_set4.sh $c $F $V || { echo "SET FAILED $V"; exit 1; }
  echo "=== card$c ${F} MHz @ ${V} mV: $(grep -A3 OD_MCLK /sys/class/drm/card$c/device/pp_od_clk_voltage | tr -s " " | tr "\n" " ")"
  e0=$(sudo -n dmesg | grep -ci "amdgpu.*\(error\|fault\|timeout\|hang\|reset\)")
  h=$(~/jt/uv/stress1.sh $dev $sec); echo "heavy: $h"
  l=$(MB_T_OVR=150 ~/jt/uv/stress1.sh $dev 40); echo "light: $l"
  e1=$(sudo -n dmesg | grep -ci "amdgpu.*\(error\|fault\|timeout\|hang\|reset\)")
  if echo "$h" | grep -q "exit 0" && echo "$l" | grep -q "exit 0" && [ "$e1" = "$e0" ]; then echo "PASS $V"; else echo "FAIL at $V (dmesg new $((e1-e0)))"; exit 3; fi
done
echo SWEEP_END

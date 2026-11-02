#!/bin/bash
# uv_validate.sh <card> <vkdev>: longer bit-exact validation of the current setting: heavy 180 s, light T=64/150 60 s
c=$1; dev=$2
echo "=== card$c validate: $(sed -n 4,13p /sys/class/drm/card$c/device/pp_od_clk_voltage | tr -s " " | tr "\n" " ")"
e0=$(sudo -n dmesg | grep -ci "amdgpu.*\(error\|fault\|timeout\|hang\|reset\)"); ok=1
for T in 4000 64 150; do s=60; [ $T = 4000 ] && s=180
  r=$(MB_T_OVR=$T ~/jt/uv/stress1.sh $dev $s); echo "T=$T: $r"; echo "$r" | grep -q "exit 0" || ok=0; done
e1=$(sudo -n dmesg | grep -ci "amdgpu.*\(error\|fault\|timeout\|hang\|reset\)")
[ $ok = 1 ] && [ "$e1" = "$e0" ] && echo "VALID card$c" || echo "INVALID card$c (dmesg new $((e1-e0)))"

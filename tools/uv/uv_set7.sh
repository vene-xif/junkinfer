#!/bin/bash
# uv_set7.sh <card> <Vlo> <V4> <V5> <V6> <V7>: like uv_set5 but a per-state voltage for sclk states 4..7 (stock clocks)
c=$1; L=$2; f=/sys/class/drm/card$c/device/pp_od_clk_voltage
w() { echo "$1" | sudo -n tee $f >/dev/null || { echo "write failed: $1"; exit 1; }; }
w "s 2 900 $L"; w "s 3 1162 $L"; w "s 4 1233 $3"; w "s 5 1275 $4"; w "s 6 1319 $5"; w "m 2 2000 $L"
w "s 7 1359 $6"; w c; w "s 7 1360 $6"; w c   # 1 MHz nudge: forces the sclk voltage upload (see uv_set5.sh)

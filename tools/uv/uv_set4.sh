#!/bin/bash
# uv_set4.sh <card> <max MHz> <mV> | reset
# AVFS is disabled by any OD voltage edit (smu7_update_avfs), so the table voltages are applied literally, and the
# memory level's VDDC entry (OD_MCLK mV) is a floor for the core voltage. This sets one voltage for sclk states 2..7
# (2 = 900 MHz, 3 = 1162 MHz, 4..7 = <max MHz>) and for the top memory level (m 2, 2000 MHz); states 0..1 stay stock.
c=$1; F=$2; V=$3; f=/sys/class/drm/card$c/device/pp_od_clk_voltage
if [ "$F" = reset ]; then echo r | sudo -n tee $f >/dev/null; echo c | sudo -n tee $f >/dev/null; exit 0; fi
w() { echo "$1" | sudo -n tee $f >/dev/null || { echo "write failed: $1"; exit 1; }; }
w "s 2 900 $V"; w "s 3 1162 $V"; for s in 4 5 6 7; do w "s $s $F $V"; done
w "m 2 2000 $V"
w c

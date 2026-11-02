#!/bin/bash
# uv_set5.sh <card> <Vlo> <Vhi> | <card> reset
# Two-level undervolt at stock clocks (any OD voltage edit disables AVFS, so these voltages apply literally):
# sclk states 2 (900) and 3 (1162) and the top memory level (m 2, 2000 MHz; its mV is the core VDDC floor) at <Vlo>,
# sclk states 4..7 (1233/1275/1319/1360) at <Vhi>. States 0..1 and memory levels 0..1 stay stock.
c=$1; f=/sys/class/drm/card$c/device/pp_od_clk_voltage
if [ "$2" = reset ]; then echo r | sudo -n tee $f >/dev/null; echo c | sudo -n tee $f >/dev/null; exit 0; fi
L=$2; H=$3
w() { echo "$1" | sudo -n tee $f >/dev/null || { echo "write failed: $1"; exit 1; }; }
w "s 2 900 $L"; w "s 3 1162 $L"; w "s 4 1233 $H"; w "s 5 1275 $H"; w "s 6 1319 $H"; w "s 7 1360 $H"
w "m 2 2000 $L"
# Kernel quirk (smu7_check_dpm_table_updated): when a memory-level voltage differs from the pptable it returns before
# looking at the sclk voltages, so sclk voltage edits are only uploaded if an sclk *clock* also changed. Nudge state 7
# by 1 MHz, commit, then restore it and commit again so every sclk level is re-uploaded with the new voltages.
w "s 7 1359 $H"; w c
w "s 7 1360 $H"
w c

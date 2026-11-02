#!/bin/bash
# Undervolt both RX 580s at boot (systemd: scripts/ji-undervolt.service, root). Values found 2026-11-01 with bit-exact
# GEMM stress sweeps (lab/mmbench MB_STRESS, ~/jt/uv/uv_sweep{4,5,6}.sh); see engineering-SIREN/current/engineering-log.md.
# Per card: sclk states 2 (900) / 3 (1162) and the top memory level (m 2, 2000 MHz - its mV is the core VDDC floor) at
# VLO, sclk states 4..7 (1233/1275/1319/1360, stock clocks) at V4..V7. Margin: >= 2 sweep steps (50 mV) above the
# first failure at that clock (card1 fails: 875 @ <=1200 MHz, 1025 @ 1360; card0: 825 low states, 975 top states).
# Notes: any OD voltage edit disables AVFS, so these voltages apply literally; and the kernel only uploads sclk voltage
# edits if an sclk clock changed too (smu7_check_dpm_table_updated returns early on a changed memory-level voltage),
# hence the 1 MHz nudge of state 7. Undo: JI_UV=off in /etc/junkinfer.env, or `echo r > pp_od_clk_voltage; echo c > ...`.
# Cards by PCI address: 0000:05:00.0 = helper (Vulkan dev 0, x4), 0000:06:00.0 = primary (Vulkan dev 1, x16).
declare -A VLO=( [0000:05:00.0]=875 [0000:06:00.0]=925 )
declare -A VTOP=( [0000:05:00.0]="1025 1025 1025 1025" [0000:06:00.0]="975 1000 1050 1075" )
[ -r /etc/junkinfer.env ] && . /etc/junkinfer.env
rc=0
for dev in /sys/bus/pci/drivers/amdgpu/0000:*; do
  pci=$(basename $dev); f=$dev/pp_od_clk_voltage
  [ -w "$f" ] || continue
  if [ "${JI_UV:-on}" = off ] || [ -z "${VLO[$pci]}" ]; then echo r > $f; echo c > $f; echo "$pci: stock"; continue; fi
  L=${VLO[$pci]}; read V4 V5 V6 H <<< "${VTOP[$pci]}"
  echo "s 2 900 $L" > $f && echo "s 3 1162 $L" > $f && echo "s 4 1233 $V4" > $f && echo "s 5 1275 $V5" > $f &&
    echo "s 6 1319 $V6" > $f && echo "m 2 2000 $L" > $f && echo "s 7 1359 $H" > $f && echo c > $f &&
    echo "s 7 1360 $H" > $f && echo c > $f || { echo "$pci: write failed"; rc=1; continue; }
  # verify the card really uses the new voltages: force state 7 then 3 briefly and read vddgfx
  lvl=$(cat $dev/power_dpm_force_performance_level)
  echo manual > $dev/power_dpm_force_performance_level
  echo 7 > $dev/pp_dpm_sclk; sleep 0.5; v7=$(cat $dev/hwmon/hwmon*/in0_input)
  echo 3 > $dev/pp_dpm_sclk; sleep 0.5; v3=$(cat $dev/hwmon/hwmon*/in0_input)
  echo $lvl > $dev/power_dpm_force_performance_level
  echo "$pci: Vlo $L Vtop $V4/$V5/$V6/$H -> measured s3 ${v3} mV, s7 ${v7} mV (level $lvl)"
  [ $((v7 - H)) -gt 15 ] || [ $((H - v7)) -gt 15 ] && { echo "$pci: state 7 voltage not applied"; rc=1; }
  [ $((v3 - L)) -gt 15 ] || [ $((L - v3)) -gt 15 ] && { echo "$pci: state 3 voltage not applied"; rc=1; }
done
exit $rc

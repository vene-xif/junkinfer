#!/usr/bin/env bash
# CPU frequency governor -> performance (AUX-1, 2026-10-23). With schedutil the host work between decode steps (expert
# cache update, the TP sync relay) ran on down-clocked cores: decode -6-9 % fresh and -20-25 % for ~1 min after a large
# prefill (schedutil 33.7 / 36.4 / 34.8 t/s vs performance 40.2 / 44.4 / 43.8 after a 17k prompt).
#   scripts/cpu_performance.sh            set it now (needs sudo)
#   scripts/cpu_performance.sh install    also at every boot (systemd unit junkinfer-cpu-performance.service)
set -euo pipefail
sudo sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done'
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
if [ "${1:-}" = install ]; then
  sudo tee /etc/systemd/system/junkinfer-cpu-performance.service > /dev/null <<'UNIT'
[Unit]
Description=JunkInfer: CPU governor performance
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done'
[Install]
WantedBy=multi-user.target
UNIT
  sudo systemctl enable junkinfer-cpu-performance.service && echo "installed (runs at boot)"
fi

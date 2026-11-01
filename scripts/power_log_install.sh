#!/bin/bash
# installs ji-power-log.service and loads the board sensor driver (it87: the B450 AORUS ELITE's IT8792E) at boot
set -e
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sudo install -m 644 "$D/ji-power-log.service" /etc/systemd/system/ji-power-log.service
echo it87 | sudo tee /etc/modules-load.d/ji-it87.conf > /dev/null
sudo modprobe it87 || true
sudo systemctl daemon-reload
sudo systemctl enable --now ji-power-log.service
systemctl is-active ji-power-log.service

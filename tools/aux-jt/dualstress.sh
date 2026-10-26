#!/bin/bash
# both cards under gpustress at increasing power caps; stops at the first card loss. usage: dualstress.sh SECS CAP_W...
S=$1; shift; cd ~/jt/gpustress
setcap() { for c in /sys/class/drm/card[0-9]; do echo "$SUDO_PW" | sudo -S -p "" sh -c "echo $(($1 * 1000000)) > $(ls -d $c/device/hwmon/hwmon*)/power1_cap"; done; }
for W in "$@"; do
  setcap $W; echo "== cap $W W ($(date +%T))"; L0=$(wc -l < ~/jt/pwr.log)
  ./gpustress 0 $S > /tmp/gs0.txt 2>&1 & A=$!; ./gpustress 1 $S > /tmp/gs1.txt 2>&1 & B=$!; wait $A $B
  tail -1 /tmp/gs0.txt; tail -1 /tmp/gs1.txt
  tail -n +$((L0 + 5)) ~/jt/pwr.log | awk -F"|" '{split($2,a," "); split($3,b," "); w0+=a[2]; w1+=b[2]; if(a[3]+0>t0)t0=a[3]+0; if(b[3]+0>t1)t1=b[3]+0; n++} END {printf "x4 avg %.0f W max %d C | x16 avg %.0f W max %d C (%d s)\n", w0/n, t0, w1/n, t1, n}'
  if [ "$(lspci | grep -c VGA)" != 2 ] || echo "$SUDO_PW" | sudo -S -p "" dmesg | grep -q "lost from bus"; then echo "CARD LOST at cap $W"; exit 1; fi
done
setcap 145; echo "all steps passed; cap back to 145"

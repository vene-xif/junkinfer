#!/bin/bash
# stress1.sh <vkdev> <seconds>: bit-exact GEMM stress on one GPU; prints result + power/clock/voltage stats of both cards during the run
dev=$1; sec=$2
t0=$(date "+%F %T")
cd ~/junkinfer/lab/mmbench
MB_DEV=$dev MB_T=${MB_T_OVR:-4000} MB_STRESS=$sec timeout $((sec+120)) ./mmbench prod_f.spv:64,prod_h.spv:32 > /tmp/stress_$dev.out 2>&1; rc=$?
t1=$(date "+%F %T")
grep -a "^stress:\|STRESS" /tmp/stress_$dev.out | tail -3
echo "exit $rc"
awk -v a="$t0" -v b="$t1" -F" [|] " "{ts=substr(\$1,1,19)} ts>=a && ts<=b {print}" ~/junkinfer/logs/power.log | python3 -c "
import sys,re
rows=[l for l in sys.stdin]
for c in (0,1):
    W=[];C=[];V=[];T=[]
    for l in rows:
        m=re.search(r\"card%d (\d+)W (\d+)C (\d+)/\d+MHz (\d+)mV\" % c, l)
        if m and int(m.group(1))>60: W.append(int(m.group(1)));T.append(int(m.group(2)));C.append(int(m.group(3)));V.append(int(m.group(4)))
    if W: print(\"card%d loaded %d samples: W avg %.0f max %d | sclk avg %.0f min %d | mV avg %.0f | temp max %d\" % (c,len(W),sum(W)/len(W),max(W),sum(C)/len(C),min(C),sum(V)/len(V),max(T)))
s=[int(re.search(r\"sum (\d+)W\",l).group(1)) for l in rows if \"sum \" in l]
if s: print(\"gpu+cpu sum max %d W\" % max(s))
"

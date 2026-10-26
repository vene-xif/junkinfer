#!/bin/bash
# job 10: decode budget = 50 % of the context (KL, 4 rows); prefill budget sweep (KL, tail as one prefill batch, decode dense); prefill t/s at 130k per budget
cd ~/jt
~/jt/spgate.sh "JI_SPARSE_FRAC=0.5" > ~/jt/spgate10a.log 2>&1
ROWS="L_code:1 L_code:2" ~/jt/spgate.sh "JI_JPF=1 JR_TAIL_B=1025 JI_SPARSE=0 JI_SPARSE_PF=32768" "JI_JPF=1 JR_TAIL_B=1025 JI_SPARSE=0 JI_SPARSE_PF=65536" "JI_JPF=1 JR_TAIL_B=1025 JI_SPARSE=0 JI_SPARSE_PF=131072" > ~/jt/spgate10b.log 2>&1
for pf in 0 8192 32768 65536 131072; do
  NMAX=0 JI_SPARSE_PF=$pf JI_CACHE_SWAPS=16 ./mtpsrv.sh >/dev/null || { echo "pf $pf: server failed"; continue; }
  echo "pf $pf: $(cd ~/junkinfer && JI_GEN=8 timeout 3000 python3 scripts/depth_bench.py 130000 8 http://127.0.0.1:8080 x nocache | grep -oE "prompt [0-9]+ tokens: prefill [0-9.]+ t/s \([0-9]+ s\)")"
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); [ -n "$P" ] && kill $P   # idle server
echo JOB10_DONE $(date +%T)

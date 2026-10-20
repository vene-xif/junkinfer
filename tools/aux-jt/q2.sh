#!/bin/bash
# queue: wait for the running spfgate, then server prefill timings with the prompt-end hint, decode KL on 2 more rows, 245k decode speed
while ps -eo args | grep -q "[s]pfgate.sh"; do sleep 20; done
for e in JI_SPF_DENSE_TAIL=0 JI_SPF_DENSE_TAIL=4096 JI_SPF_DENSE_TAIL=8192; do ~/jt/restart.sh $e >/dev/null; echo -n "$e: "; ~/jt/venv/bin/python ~/jt/srvpf.py ~/jt/docs_all.txt; done
for e in "JI_X=0" "JI_SP_NORM=2 JI_SP_BG=2"; do
  ~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache $e >/dev/null; echo "== $e"; (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 245000 3 2>&1 | grep decode | cut -c1-90)
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}"); kill $P; sleep 5
for e in "JI_SP_NORM=2 JI_SP_BG=2 JI_SP_VARK=0.35" "JI_SP_NORM=2 JI_SP_BG=2 JI_SP_VARK=0.7"; do ~/jt/kl.sh L_code 1 $e; done
for row in "L_code98 0" "L_code 2"; do for e in "JI_X=0" "JI_SP_NORM=2 JI_SP_BG=2"; do ~/jt/kl.sh $row $e; done; done

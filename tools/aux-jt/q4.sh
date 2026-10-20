#!/bin/bash
for e in "JI_X=0" "JI_SP_NORM=2 JI_SP_BG=2" "JI_SP_BG=2"; do
  ~/jt/restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache $e >/dev/null; echo "== $e"; (cd ~/junkinfer && JI_SLOT_DIR=/mnt/ssd/kvcache scripts/depth_kv.sh 245000 3 2>&1 | grep decode | cut -c1-90)
done
P=$(ps -eo pid,args | grep "[b]in/llama-server -m /mnt/ssd" | awk "{print \$1}"); kill $P; sleep 5
~/jt/kl.sh L_code98 0 JI_SP_BG=8
~/jt/kl.sh L_code 2 JI_SP_BG=8
cd ~ && ROW=1 DEC=default ~/jt/spfgate.sh "JI_X=0"

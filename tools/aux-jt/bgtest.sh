#!/bin/bash
for cfg in "JI_SPARSE=4096" "JI_SPARSE=4096 JI_SP_BG=1" "JI_SPARSE=4096 JI_SP_BG=2"; do
  ~/jt/kl.sh L_code 0 $cfg; cp ~/jt/kl.log ~/jt/kl_$(echo $cfg | tr " =" "__").log
done

#!/bin/bash
for cfg in "JI_SPARSE=8192 JI_SP_BG=2 JI_SP_SEL=1" "JI_SPARSE=4096 JI_SP_BG=2"; do
  ~/jt/kl.sh L_code 0 $cfg; cp ~/jt/kl.log ~/jt/kl_c_$(echo $cfg | tr " =" "__").log
done

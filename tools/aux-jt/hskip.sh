#!/bin/bash
# rank 1 (helper) mixer / moe busy with one kernel skipped on the helper (JI_TP_SKIP): TP prefill at 2k depth
cd ~/jt
for k in ${KS:-none pf_gdn_cstate pf_gdn_conv pf_gdn_out pf_f2h pf_add2h pf_rms pf_qk_q8 pf_attn_out_q8 pf_fa_p4 pf_gdn_rule}; do
  ./restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache JI_TP=2 JI_JPF=1 JI_TP_MARGIN_MIB=1500 JI_PROFILE=1 JI_PF_TL=1 JI_TP_SKIP=$k > /dev/null
  timeout 300 python3 dpb.py 2000 > /dev/null 2>&1; timeout 300 python3 dpb.py 2000 > /dev/null 2>&1
  echo "$k $(grep -a "TL rank 1: 4092" srv.log | tail -1 | grep -o "busy [0-9.]* (mixer [0-9.]*, moe [0-9.]*")"
done

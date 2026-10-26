#!/bin/bash
# kernel-family cost by repetition (JI_PF_REP=prefix:N): TP prefill at 2k depth, TL busy per rank
cd ~/jt
for pre in ${PRES:-none pf_mm_id pf_mm pf_prep pf_f32mm pf_gdn_rule pf_gdn_conv pf_topk pf_rowmap pf_silu pf_combine pf_rms pf_qk pf_fa pf_f2h pf_add2h pf_attn pf_gdn_out pf_gdn_cst}; do
  ./restart.sh JI_SLOT_DIR=/mnt/ssd/kvcache ${CFG:-JI_TP=2 JI_JPF=1 JI_TP_MARGIN_MIB=1500} JI_PROFILE=1 JI_PF_TL=1 JI_PF_REP=$pre:${N:-3} $EXTRA > /dev/null
  timeout 600 python3 dpb.py ${DEPTH:-2000} > /dev/null 2>&1
  echo "$pre $(grep -a "TL rank 0: ${TLN:-4092}" srv.log | tail -1 | sed "s/.*busy \([0-9.]*\).*/\1/") $(grep -a "TL rank 1: ${TLN:-4092}" srv.log | tail -1 | sed "s/.*busy \([0-9.]*\).*/\1/")"
done

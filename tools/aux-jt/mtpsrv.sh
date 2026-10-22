#!/bin/bash
# TP2 production config + MTP head on the helper (JDE); NMAX drafts; extra env passes through; local only
P=$(ps -eo pid,args | grep "[b]in/llama-server -m " | awk "{print \$1}"); if [ -n "$P" ]; then kill $P; for i in $(seq 1 60); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -9 $P 2>/dev/null; sleep 1; fi
cd ~/junkinfer
JI_TP=2 JI_JPF=${JI_JPF:-1} JI_TP_MARGIN_MIB=${JI_TP_MARGIN_MIB:-1900} JI_SLOT_DIR=/mnt/ssd/kvcache JI_CTX=${CTX:-262144} \
  setsid nohup scripts/serve.sh /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf \
  $( [ "${NMAX:-1}" = 0 ] || echo "-md /mnt/ssd/models/mtp-IQ4_NL.v2.gguf --spec-type draft-mtp -ngld 0 -devd none --no-spec-draft-backend-sampling --spec-draft-n-max ${NMAX:-1} --spec-draft-p-min ${PMIN:-0}" ) ${SARGS:-} > ~/jt/mtp_srv.log 2>&1 < /dev/null &
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8080/health >/dev/null && echo up && exit 0; sleep 5; done; echo fail

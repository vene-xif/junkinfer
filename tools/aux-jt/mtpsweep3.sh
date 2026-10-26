#!/bin/bash
# usage: mtpsweep2.sh "LABEL ENV..." ...   each arg: label followed by env assignments for mtpsrv.sh
for cfg in "$@"; do
  set -- $cfg; label=$1; shift
  env "$@" JI_TP_MARGIN_MIB=${MARGIN:-1900} JI_PROFILE=100 SARGS="-lv 4" ./mtpsrv.sh >/dev/null || { echo "$label: server failed"; continue; }
  echo "== $label"; python3 spectest.py | cut -c1-100
  tr -d "\000" < mtp_srv.log | grep -a -E "token steps 100" | tail -2 | cut -c14-140
done

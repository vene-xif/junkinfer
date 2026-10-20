#!/bin/bash
# sparse prefill KL gate: each config prefills the prefix itself (own empty state dir); decode dense unless DEC=default
ROW=${ROW:-0}; SET=${SET:-L_code}
DECARG="JI_SPARSE=0"; [ "${DEC:-dense}" = default ] && DECARG="JI_X=1"
for cfg in "$@"; do
  tag=$(echo "$cfg" | tr " =" "__"); d=/mnt/ssd/kvcache/jref_spf/${SET}_${ROW}_$tag; rm -rf "$d"; mkdir -p "$d"
  ~/jt/kl.sh $SET $ROW $DECARG JR_STATE_DIR=$d $cfg
  rm -rf "$d"
done

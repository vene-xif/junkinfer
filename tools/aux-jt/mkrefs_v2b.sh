#!/bin/bash
# v2 full-attention references + prefix states (2026-10-23): GGML_VK_SUBALLOCATION_BLOCK_SIZE 1 GiB - with 3.75 GiB
# blocks the graph path lost the first attention layer's KV writes beyond ~8k cells (all v1 refs / jref states corrupt)
export GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES-1}
cd ~/junkinfer/engine
M="-m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -ngl 99 --cpu-moe --no-repack --load-mode none -fa on --fit off -c 262144 -ctk q8_0 -ctv q8_0 -t 6 -b 4096 -ub 4096"
export JI_SPARSE=0 JI_SPARSE_PF=0 JI_JPF=1 JI_DECODE=1 JI_CACHE_MIB=auto GGML_VK_SUBALLOCATION_BLOCK_SIZE=1073741824 RADV_PERFTEST=nogttspill JR_STATE_DIR=/mnt/ssd/kvcache/jref_v2 JR_TAIL_B=4
mkdir -p $JR_STATE_DIR
for sr in L_code:1 L_code:2; do
  set=${sr%:*}; r=${sr#*:}
  out=~/jt/${set}_r${r}_v2.jref
  [ -s $out ] && continue
  echo "$(date +%T) $set r$r"
  build/bin/llama-junk-ref $M --jr-mode dump --jr-tokens ~/junkinfer/data/corpus/$set.jtok --jr-rows $r:$((r+1)) --jr-out $out 2>&1 | grep -E "SUMMARY|rror" | cut -c1-200
  python3 ~/jt/stzero.py $JR_STATE_DIR/$(ls -t $JR_STATE_DIR | head -1) 2>/dev/null | cut -c1-120
done
echo MKREFS_V2_DONE $(date +%T)

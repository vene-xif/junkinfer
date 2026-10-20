#!/bin/bash
# KL vs the full-attention reference of a long row: kl.sh SET ROW [env assignments...]
cd ~/junkinfer/engine
SET=$1; R=$2; shift 2
M="-m ${MODEL:-/mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf} -ngl 99 --cpu-moe --no-repack --load-mode none -fa on --fit off -c 262144 -ctk q8_0 -ctv q8_0 -t 6 -b 4096 -ub 4096"
env JI_DECODE=1 JI_CACHE_MIB=auto GGML_VK_SUBALLOCATION_BLOCK_SIZE=4026531840 RADV_PERFTEST=nogttspill JR_STATE_DIR=/mnt/ssd/kvcache/jref JR_TAIL_B=4 "$@" \
  build/bin/llama-junk-ref $M --jr-mode compare --jr-tokens ~/junkinfer/data/corpus/$SET.jtok --jr-rows $R:$((R+1)) --jr-ref ${JREF:-~/jt/${SET}_r${R}_full.jref} > ~/jt/kl.log 2>&1
python3 ~/jt/klsum.py "$SET r$R $*"

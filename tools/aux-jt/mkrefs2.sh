#!/bin/bash
export GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES-1}   # x16 card (AUX-1 has 2 GPUs since 2026-10-20)
# full-attention references (dense: JI_SPARSE=0) for L_code rows 0-2 with the current kernels; prefix states cached
cd ~/junkinfer/engine
export JI_SPARSE=0 JI_SPARSE_PF=0 JI_DECODE=1 JI_CACHE_MIB=auto GGML_VK_SUBALLOCATION_BLOCK_SIZE=4026531840 RADV_PERFTEST=nogttspill JR_STATE_DIR=/mnt/ssd/kvcache/jref JR_TAIL_B=4
for r in 0 1 2; do
  build/bin/llama-junk-ref -m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -ngl 99 --cpu-moe --no-repack --load-mode none -fa on --fit off -c 262144 -ctk q8_0 -ctv q8_0 -t 6 -b 4096 -ub 4096 --jr-mode dump --jr-tokens ~/junkinfer/data/corpus/L_code.jtok --jr-rows $r:$((r+1)) --jr-out ~/jt/L_code_r${r}_full.jref 2>&1 | grep -E "SUMMARY" | cut -c1-200
done
echo MKREFS2_DONE

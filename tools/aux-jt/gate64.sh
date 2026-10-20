#!/bin/bash
# 64k gate: dense reference (JI_SPARSE=0) of L_code64 row 0, then sparse+tail KL at 64k (JI_SP_MIN lowered)
cd ~/junkinfer/engine
M="-m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -ngl 99 --cpu-moe --no-repack --load-mode none -fa on --fit off -c 262144 -ctk q8_0 -ctv q8_0 -t 6 -b 4096 -ub 4096"
export JI_DECODE=1 JI_CACHE_MIB=auto GGML_VK_SUBALLOCATION_BLOCK_SIZE=4026531840 RADV_PERFTEST=nogttspill JR_STATE_DIR=/mnt/ssd/kvcache/jref JR_TAIL_B=4
JI_SPARSE=0 build/bin/llama-junk-ref $M --jr-mode dump --jr-tokens ~/junkinfer/data/corpus/L_code64.jtok --jr-rows 0:1 --jr-out ~/jt/L_code64_r0_full.jref 2>&1 | grep -E "SUMMARY" | cut -c1-200
for B in 8192 16384; do JREF=~/jt/L_code64_r0_full.jref ~/jt/kl.sh L_code64 0 JI_SPARSE=$B JI_SP_MIN=32768; done
echo GATE64_DONE

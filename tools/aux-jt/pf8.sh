#!/bin/bash
export GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES-1}   # x16 card (AUX-1 has 2 GPUs since 2026-10-20)
# prefill perf-logger run on the 8k prompt (ub 4096); log in ~/jt/pf8.log
cd ~/junkinfer/engine
env "$@" GGML_VK_PERF_LOGGER=1 JI_DECODE=1 JI_CACHE_MIB=auto GGML_VK_SUBALLOCATION_BLOCK_SIZE=4026531840 RADV_PERFTEST=nogttspill timeout 1200 build/bin/llama-completion -m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -ngl 99 --cpu-moe --no-repack --load-mode none -fa on --fit off -c 16384 -ctk q8_0 -ctv q8_0 -t 6 -b 4096 -ub 4096 -f ~/jt/p8k.txt -n 1 -no-cnv --temp 0 > ~/jt/pf8.log 2>&1
grep -E "prompt eval" ~/jt/pf8.log

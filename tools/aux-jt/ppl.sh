#!/bin/bash
# quick prefill-path correctness check: perplexity of p20k.txt, 2 chunks of 4096
cd ~/junkinfer/engine
env "$@" RADV_PERFTEST=nogttspill GGML_VK_SUBALLOCATION_BLOCK_SIZE=4026531840 build/bin/llama-perplexity -m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -ngl 99 --cpu-moe --no-repack --load-mode none -fa on --fit off -c 4096 -b 4096 -ub 4096 -ctk q8_0 -ctv q8_0 -t 6 -f ~/jt/p20k.txt --chunks 2 2>&1 | grep -E "^\[|Final estimate"

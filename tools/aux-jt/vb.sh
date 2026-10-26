#!/bin/bash
# verify-step cost (1..4 tokens) with the server settings; env passes through (JI_TP=2 etc.)
O=$(pgrep -f "^(/home/vene/junkinfer/engine/)?build/bin/llama-|^/home/vene/junkinfer/engine/build/bin/llama-")
if [ -n "$O" ]; then echo "refusing: another model process is running ($O)"; exit 1; fi
sync; echo "$SUDO_PW" | sudo -S -p "" sh -c "echo 3 > /proc/sys/vm/drop_caches"
export RADV_PERFTEST=nogttspill GGML_VK_SUBALLOCATION_BLOCK_SIZE=4026531840 JI_DECODE=1 JI_CACHE_MIB=auto JI_CACHE_SWAPS=2
DEV="-dev Vulkan1"
if [ "${JI_TP:-0}" != 0 ]; then export GGML_VK_HOST_DEVICE=1 JI_TP_HOT=${JI_TP_HOT:-$HOME/junkinfer/ref/tp/tp_hot_qwen36_exl3.bin}; else export GGML_VK_VISIBLE_DEVICES=1; DEV=""; fi
VB_FILE=${VB_FILE:-$HOME/jt/p8k.txt} ~/junkinfer/engine/build/bin/llama-junk-vbench -m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf \
  -ngl 99 $DEV --cpu-moe --no-repack --load-mode none -fa on --fit off -c ${CTX:-262144} -ctk q8_0 -ctv q8_0 -t 6 -b 4096 -ub 4096 ${VB_ARGS:-} 2>&1 | grep -E "${VB_GREP:-vbench|error|failed}"

#!/bin/bash
# dual-card prefill probe: llama-completion prompt eval on a prompt file. usage: dpf.sh PROMPT CTX label args...
P=$1; C=$2; L=$3; shift 3
cd ~/junkinfer/engine
sync; echo "$SUDO_PW" | sudo -S -p "" sh -c "echo 3 > /proc/sys/vm/drop_caches"
env JI_DECODE=0 GGML_VK_SUBALLOCATION_BLOCK_SIZE=4026531840 RADV_PERFTEST=nogttspill timeout 3000 build/bin/llama-completion \
  -m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf --no-repack --load-mode none -fa on --fit off -c $C -ctk q8_0 -ctv q8_0 -t 6 \
  -f $P -n 1 -no-cnv --temp 0 "$@" > ~/jt/dpf.log 2>&1
echo "$L: $(grep -a "prompt eval time" ~/jt/dpf.log | grep -o "[0-9.]* tokens per second")  $(grep -a -c "error\|failed" ~/jt/dpf.log) errs"
grep -a "model buffer size\|KV buffer size\|compute buffer size" ~/jt/dpf.log | sed 's/^.*I //' | cut -c1-80 | tr '\n' ';'; echo

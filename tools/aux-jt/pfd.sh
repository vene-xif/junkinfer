#!/bin/bash
# prefill of the 8k prompt (llama-completion): DUAL=S (layers S..39 on the x4 card) PIPE=1 (pipeline parallel), UB, B, CTX, PF
O=$(pgrep -f "^(/home/vene/junkinfer/engine/)?build/bin/llama-|^/home/vene/junkinfer/engine/build/bin/llama-")
if [ -n "$O" ]; then echo "refusing: another model process is running ($O)"; exit 1; fi
cd ~/junkinfer/engine
UB=${UB:-2048}; B=${B:-8192}
if [ -n "${TP:-}" ]; then
  export GGML_VK_HOST_DEVICE=1 JI_TP=$TP JI_TP_HOT=${JI_TP_HOT:-$HOME/junkinfer/ref/tp/tp_hot_qwen36_exl3.bin}; unset GGML_VK_VISIBLE_DEVICES
  DEV=(-dev Vulkan1 --cpu-moe)
elif [ -n "${DUAL:-}" ]; then
  [ -n "${HOSTDEV-1}" ] && export GGML_VK_HOST_DEVICE=${HOSTDEV-1}; unset GGML_VK_VISIBLE_DEVICES
  DEV=(-dev Vulkan1,Vulkan0 -ts "$DUAL,$((41 - DUAL))" ${OT:+-ot "$OT"} --n-cpu-moe "${NCMOE:-$DUAL}")
else export GGML_VK_VISIBLE_DEVICES=1; DEV=(--cpu-moe); fi
env "$@" JI_PIPELINE=${PIPE:-0} JI_DECODE=1 JI_CACHE_MIB=auto GGML_VK_SUBALLOCATION_BLOCK_SIZE=4026531840 RADV_PERFTEST=nogttspill timeout 1500 build/bin/llama-completion -m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -ngl 99 "${DEV[@]}" --no-repack --load-mode none -fa on --fit off -c ${CTX:-32768} -ctk q8_0 -ctv q8_0 -t 6 -b $B -ub $UB -f ${PF:-$HOME/jt/p8k.txt} -n ${NGEN:-1} -no-cnv --temp 0 ${EXTRA:-} > ~/jt/pfd.log 2>&1
echo "DUAL=${DUAL:-} PIPE=${PIPE:-0} UB=$UB B=$B $*: $(grep -o "prompt eval time.*" ~/jt/pfd.log | grep -o "[0-9.]* tokens per second") | $(grep -o "pipeline parallelism enabled\|retrying without pipeline" ~/jt/pfd.log | head -1) $(grep -i "error\|failed\|abort" ~/jt/pfd.log | head -2)"

#!/bin/bash
# KL vs the full-attention reference of a long row: kl.sh SET ROW [env assignments...]
# DUAL=S: both cards, layers 0..S-1 on the x16 card (as serve.sh JI_DUAL); UB = ubatch (default 4096)
DEV=(--cpu-moe)
if [ -n "${TP:-}" ]; then   # tensor-parallel MoE helper on the x4 card (JI_TP)
  export GGML_VK_HOST_DEVICE=1 JI_TP=$TP JI_TP_FRAC=${JI_TP_FRAC:-0.7} JI_TP_HOT=${JI_TP_HOT:-/home/vene/jt/route_counts.bin}; unset GGML_VK_VISIBLE_DEVICES
  DEV=(-dev Vulkan1 --cpu-moe)
elif [ -n "${DUAL:-}" ]; then
  export GGML_VK_HOST_DEVICE=1; unset GGML_VK_VISIBLE_DEVICES
  DEV=(-dev Vulkan1,Vulkan0 -ts "$DUAL,$((41 - DUAL))" --n-cpu-moe "$DUAL")
else
  export GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES-1}   # x16 card (AUX-1 has 2 GPUs since 2026-10-20)
fi
cd ~/junkinfer/engine
SET=$1; R=$2; shift 2
UB=${UB:-4096}
M="-m ${MODEL:-/mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf} -ngl 99 --no-repack --load-mode none -fa on --fit off -c 262144 -ctk q8_0 -ctv q8_0 -t 6 -b ${B:-4096} -ub $UB"
env JI_DECODE=1 JI_CACHE_MIB=auto GGML_VK_SUBALLOCATION_BLOCK_SIZE=4026531840 RADV_PERFTEST=nogttspill JR_STATE_DIR=/mnt/ssd/kvcache/jref JR_TAIL_B=4 "$@" \
  build/bin/llama-junk-ref $M "${DEV[@]}" --jr-mode compare --jr-tokens ~/junkinfer/data/corpus/$SET.jtok --jr-rows $R:$((R+1)) --jr-ref ${JREF:-~/jt/${SET}_r${R}_full.jref} ${KLX:-} > ~/jt/kl.log 2>&1
python3 ~/jt/klsum.py "$SET r$R ${DUAL:+dual$DUAL ub$UB }${TP:+tp }$*"

#!/bin/bash
export GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES-1}   # x16 card (AUX-1 has 2 GPUs since 2026-10-20)
# draft acceptance per MTP sidecar quantization (draft n = 3), 32k context to leave VRAM for the draft
O=$(ps -eo pid,args | grep -E "[b]in/llama-(junk-ref|completion|perplexity|bench|cli)" | awk "{print \$1}")
if [ -n "$O" ]; then echo "refusing: another model process is running"; exit 1; fi
for V in "$@"; do
  P=$(ps -eo pid,args | grep "[b]in/llama-server" | awk '{print $1}'); [ -n "$P" ] && kill $P && sleep 5
  export JI_DECODE=1 JI_CACHE_MIB=auto JI_CACHE_SWAPS=2 GGML_VK_SUBALLOCATION_BLOCK_SIZE=4026531840 RADV_PERFTEST=nogttspill
  nohup ~/junkinfer/engine/build/bin/llama-server -m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -ngl 99 --cpu-moe --no-repack \
    --load-mode none -fa on --fit off -c 32768 -ctk q8_0 -ctv q8_0 -np 1 --jinja -t 6 -b 4096 -ub 4096 --host 127.0.0.1 --port 8080 \
    $( [ "$V" = none ] && echo "" || echo "-md /mnt/ssd/models/$V --spec-type draft-mtp --spec-draft-n-max ${NMAX:-3} --spec-draft-n-ubatch 256 -ngld 99 -cmoed ${SPEC_EXTRA:-}" ) -v > ~/jt/mtp_srv.log 2>&1 &
  for i in $(seq 1 100); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 5; done
  python3 ~/jt/mtpacc.py $V
  grep -a "statistics" ~/jt/mtp_srv.log | tail -1 | grep -o "acc rate/pos = ([^)]*)" | sed "s/^/  $V /"
done
P=$(ps -eo pid,args | grep "[b]in/llama-server" | awk '{print $1}'); [ -n "$P" ] && kill $P

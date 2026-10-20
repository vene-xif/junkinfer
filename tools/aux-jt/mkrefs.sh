#!/bin/bash
# full-attention references (EXL3 4.0 bpw, q8_0 head-major KV, JDE 4-token tail) for the long rows; prefix states cached
cd ~/junkinfer/engine
M="-m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -ngl 99 --cpu-moe --no-repack --load-mode none -fa on --fit off -c 262144 -ctk q8_0 -ctv q8_0 -t 6 -b 4096 -ub 4096"
export JI_SPARSE=0 JI_SPARSE_PF=0 JI_DECODE=1 JI_CACHE_MIB=auto GGML_VK_SUBALLOCATION_BLOCK_SIZE=4026531840 RADV_PERFTEST=nogttspill JR_STATE_DIR=/mnt/ssd/kvcache/jref JR_TAIL_B=4
for set in L_code L_prose; do for r in 0 1 2; do
  out=~/jt/${set}_r${r}_full.jref
  [ -s $out ] && continue
  [ $set = L_code ] && [ $r = 0 ] && cp ~/jt/Lcode32_full.jref $out && continue
  build/bin/llama-junk-ref $M --jr-mode dump --jr-tokens ~/junkinfer/data/corpus/$set.jtok --jr-rows $r:$((r+1)) --jr-out $out 2>&1 | grep -E "SUMMARY" | cut -c1-200
done; done
echo MKREFS_DONE

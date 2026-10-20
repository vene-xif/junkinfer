#!/bin/bash
# attn_d2 variants as whole shader files: microbench 116k/245k + max VGPR
cd ~/junkinfer/engine
f=src/junkinfer/shaders/attn_d2.comp
cp $f ~/jt/d2.comp
M="-m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -ngl 99 --cpu-moe --no-repack --load-mode none -fa on --fit off -c 4096 -ctk q8_0 -ctv q8_0 -t 6 -p Hi -n 1 -no-cnv"
for v in "$@"; do
  cp $v $f; touch $f
  cmake --build build -j12 --target llama-completion 2>&1 | grep -E "error" -A3 | head -5
  [ -n "$NOISA" ] || JI_ATTN_TEST=245000 JI_CACHE_MIB=500 JI_DECODE=1 RADV_DEBUG=shaders,nocache timeout 600 build/bin/llama-completion $M > /dev/null 2>~/jt/isa2.txt
  echo -n "$(basename $v): $([ -n "$NOISA" ] || python3 ~/jt/isa.py ~/jt/isa2.txt s_buffer_load_dwordx4 8 | grep -o 'max v[0-9]*' | tr '\n' ' ') "
  for n in 116000 245000; do JI_CACHE_MIB=500 JI_ATTN_TEST=$n JI_DECODE=1 RADV_PERFTEST=nogttspill timeout 600 build/bin/llama-completion $M -v 2>&1 | grep "jde_attn_test" | python3 -c "
import sys, re
for l in sys.stdin:
    m = re.search(r'n_kv (\d+): ([0-9.]+) us', l)
    if m: print(m.group(1) + ':' + m.group(2) + 'us', end=' ')
    m = re.search(r'max \|f16 - q8\| = ([^ ]+)', l)
    if m: print('err ' + m.group(1), end=' ')
"; done; echo
done
cp ~/jt/d2.comp $f; touch $f
cmake --build build -j12 --target llama-completion llama-server 2>&1 | grep -E "error" | head -3

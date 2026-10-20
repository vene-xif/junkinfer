#!/bin/bash
# x3_mv probes: JI_BENCH timings with x3_mv / x3_mvq compiled with -DX3_EXP=$1 (0 = production)
cd ~/junkinfer/engine
D=~/jt/x3exp$1; rm -rf $D; cp -r build/bin/jde-shaders $D
for b in 4 6; do
  glslc --target-env=vulkan1.2 -O -DBITS=$b -DNT=1 -DX3_EXP=$1 src/junkinfer/shaders/x3_mv.comp -o $D/x3_mv_${b}_1.spv
  glslc --target-env=vulkan1.2 -O -DBITS=$b -DNT=1 -DFUSEQ -DX3_EXP=$1 src/junkinfer/shaders/x3_mv.comp -o $D/x3_mvq_${b}_1.spv
done
JI_SHADER_DIR=$D JI_BENCH=1 JI_DECODE=1 JI_CACHE_MIB=64 RADV_PERFTEST=nogttspill timeout 600 build/bin/llama-completion -m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -ngl 99 --cpu-moe --no-repack --load-mode none -fa on --fit off -c 4096 -ctk q8_0 -ctv q8_0 -t 6 -p "Hi" -n 1 -no-cnv --temp 0 -v 2>&1 | grep "jde_bench: exl3\|jde_bench: output" | sed "s/.*jde_bench: /EXP$1 /"

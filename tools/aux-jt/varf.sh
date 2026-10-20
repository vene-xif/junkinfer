#!/bin/bash
# FA2 variants as whole shader files: varf.sh /tmp/var/a.comp ... ; 8k FA time each
cd ~/junkinfer/engine
f=ggml/src/ggml-vulkan/vulkan-shaders/ji_fa2_prefill.comp
cp $f ~/jt/fa2.comp
for v in "$@"; do
  cp $v $f; touch $f
  cmake --build build -j12 --target llama-completion 2>&1 | grep -E "error" -A3 | head -5
  ~/jt/fasweep.sh ~/jt/p8k.txt 16384 2048 "" | sed "s#^default#$(basename $v)#"
done
cp ~/jt/fa2.comp $f; touch $f
cmake --build build -j12 --target llama-completion llama-server 2>&1 | grep -E "error" | head -3

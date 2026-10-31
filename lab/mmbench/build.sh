#!/bin/bash
# mmbench + the GEMM variants: <name>_f.spv (full, 64-row tiles) / <name>_h.spv (half, 32-row tiles)
# prod = the engine's pf_mm_id / pf_mm_id32 (copied pf_mm.comp); further variants from mm2.comp when present
set -e; cd "$(dirname "$0")"
g++ -O2 -std=c++17 mmbench.cpp -o mmbench -lvulkan
G="glslc --target-env=vulkan1.2 -O -DBITS=4 -DX3F -DMMID"
$G -DBN64 -DBK16 pf_mm.comp -o prod_f.spv
$G -DBK16 pf_mm.comp -o prod_h.spv
$G -DBN64 pf_mm.comp -o bk32_f.spv
$G pf_mm.comp -o bk32_h.spv
G2="glslc --target-env=vulkan1.2 -O -DBITS=4 -DX3F -DMMID -DBK16"
$G2 -DBN16 mm2.comp -o n16.spv
$G2 -DBN8 mm2.comp -o n8.spv
$G2 mm2.comp -o n32.spv
$G2 -DBN64 mm2.comp -o n64.spv
for a in NODEC NOFMA NOEPI; do $G2 -DBN8 -DABL_$a mm2.comp -o n8_$a.spv; $G2 -DBN64 -DABL_$a mm2.comp -o n64_$a.spv; done
G3="glslc --target-env=vulkan1.2 -O -DBITS=4"
for nt in 4 8 16; do for sv in 2 4 8; do for pf in 0 1; do
  $G3 -DNT=$nt -DS=$sv -DPF=$pf mv_id.comp -o mv${nt}_s${sv}_p${pf}.spv; done; done; done
echo built

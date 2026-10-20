#!/bin/bash
# FA kernel time + prompt eval on a prompt file (EXL3) for each env config string; usage: fasweep.sh PROMPT CTX UB "cfg" ...
cd ~/junkinfer/engine
P=$1; C=$2; U=$3; shift 3
for cfg in "$@"; do
  env $cfg GGML_VK_PERF_LOGGER=1 GGML_VK_SUBALLOCATION_BLOCK_SIZE=4026531840 RADV_PERFTEST=nogttspill timeout 1800 build/bin/llama-completion -m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -ngl 99 --cpu-moe --no-repack --load-mode none -fa on --fit off -c $C -ctk q8_0 -ctv q8_0 -t 6 -b $U -ub $U -f $P -n 32 -no-cnv --temp 0 > ~/jt/fs.out 2> ~/jt/fs.err
  python3 - "$cfg" <<'PY'
import re, sys, hashlib, os
fa = 0.0
err = open(os.path.expanduser("~/jt/fs.err")).read().splitlines()
for l in err:
    m = re.match(r"(.+?): (\d+) x ([0-9.]+) us = ([0-9.]+) us", l.strip())
    if m and m.group(1).startswith("FLASH_ATTN"): fa += float(m.group(4))
txt = open(os.path.expanduser("~/jt/fs.out"), "rb").read()
pe = [l for l in err if "prompt eval time" in l]
print(f"{sys.argv[1] or 'default':30s} FA {fa/1e3:7.0f} ms  out {hashlib.md5(txt).hexdigest()[:8]} len {len(txt)}  {pe[-1].split('=')[1].strip()[:50] if pe else 'NO PROMPT EVAL'}")
PY
done

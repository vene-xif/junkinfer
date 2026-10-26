#!/bin/bash
# prefill bench on the running server: 8k, ~80k, optionally 245k (PFB_FULL=1)
cd ~/jt
echo -n "8k:   "; python3 srvpf.py p8k.txt
echo -n "8k:   "; python3 srvpf.py p8k.txt
echo -n "80k:  "; python3 srvpf.py p68k.txt
if [ -n "${PFB_FULL:-}" ]; then echo -n "245k: "; (cd ~/junkinfer && JI_PROMPT_FILE=/mnt/ssd/kvcache/depth_245000.txt python3 scripts/big_bench.py 245000 8 | tail -1); fi

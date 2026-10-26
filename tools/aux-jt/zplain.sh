#!/bin/bash
# zplain.sh TAG CTX UB [extra args]: plain llama.cpp server (no JDE), 32k-doc prompt, slot save, zero-row report
T=$1 C=$2 U=$3; shift 3
cd ~/junkinfer
RADV_PERFTEST=nogttspill GGML_VK_SUBALLOCATION_BLOCK_SIZE=${SUB:-4026531840} GGML_VK_VISIBLE_DEVICES=1 setsid nohup ${BIN:-engine/build/bin/llama-server} -m /mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf -ngl 99 --cpu-moe --no-repack --load-mode none -fa on --fit off -c $C -ctk q8_0 -ctv q8_0 -np 1 --jinja -t 6 -b $U -ub $U --port 8080 --slot-save-path /mnt/ssd/kvcache "$@" > ~/jt/plain_srv.log 2>&1 < /dev/null &
SP=$!
for i in $(seq 120); do curl -sf localhost:8080/health >/dev/null && break; sleep 5; done
JI_TEMP=0 timeout 1200 python3 scripts/depth_bench.py 32000 4 http://127.0.0.1:8080 x nocache | tail -1
curl -s -X POST localhost:8080/slots/0?action=save -H "Content-Type: application/json" -d "{\"filename\":\"zchk_$T.bin\"}" >/dev/null
echo "$T: $(python3 ~/jt/stzero.py /mnt/ssd/kvcache/zchk_$T.bin)"
kill $SP; for i in $(seq 30); do kill -0 $SP 2>/dev/null || break; sleep 1; done

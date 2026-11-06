# Engine comparison: stock llama.cpp vs JunkInfer (AUX-1), started 2026-11-05 — INCOMPLETE

Goal (user): stock llama.cpp (1 and 2 cards) vs JunkInfer (1 card, 2 cards best config: EXL3 4.0 bpw, KVarN-6, TP2,
MTP) on Qwen3.6-35B-A3B, contexts up to ~32k, **3072-token decodes** (expert caches need that long to warm up).
Stock uses the closest normal GGUF available, `Qwen3.6-35B-IQ4_NL.gguf` (~4.5 bpw, on AUX-1 in
`~/models/ssd-overflow`), q8_0 KV, `--moe-cache-mib` hot-expert cache, mainline 8bb3cf0 (`~/llama.cpp-master`).

Files: `cmp_bench.py` (client: 3 spectest prompts + ~8k/16k/32k prompts, 3072 tokens each, cache_prompt off),
`cmp_run.sh` (runner; configs stock1 stock2 ji1 ji2nomtp ji2 — **update the stock lines with the tuned -ncmoe /
--moe-cache-mib below before running**), `tune_stock.sh` (stock placement tuning), on AUX-1 in `~/cmp/`.
Stop `junkinfer.service` first; restart it afterwards.

## Stock placement tuning (3072-token story decode + 5448-token prefill, -c 40960, q8_0 KV, -ub 2048)
See `tune_stock_2026-11-05.log`. Best: **1 card: -ncmoe 40 --moe-cache-mib 5000 -> 18.8 t/s decode, 229 t/s
prefill; 2 cards: -ncmoe 40 --moe-cache-mib 11000 -> 23.7 t/s decode, 109 t/s prefill** (2 cards with experts
split across GPUs: 5.6-7.6 t/s, prefill ~160). Note: the same 1-card config decoded 23.2 t/s over 512 tokens: the
stock cache hit rate falls over longer generations. Odd outlier: 1 card -ncmoe 30 / 1500 MiB = 2.6 t/s (VRAM
over-commit suspected, not checked).

## Status
Tuning done; the comparison runs themselves (`cmp_run.sh`) have NOT been run yet.

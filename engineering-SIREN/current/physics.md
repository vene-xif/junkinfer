# Physics — what this hardware can do with this model

Status: ACTIVE. Derived 2026-10-15. Re-derive when the hardware or the model changes.
Every speed claim must sit below these ceilings. A measurement above a ceiling is a bug.

## The model: Qwen3.6-35B-A3B (`qwen35moe`)

Source: GGUF metadata + HF config.json (2026-10-15).

| item | value |
|---|---|
| layers | 40: 30 Gated DeltaNet (fixed-size state) + 10 gated softmax attention (every 4th) + 1 MTP layer |
| hidden | 2048 |
| attention | 16 query heads, 2 KV heads, head dim 256, partial RoPE (64 of 256 dims), output gate |
| DeltaNet | 16 QK heads x 128, 32 V heads x 128, conv kernel 4 |
| MoE | 256 experts, top-8, expert width 512, plus one shared expert (width 512) per layer |
| vocab | 248,320 |
| expert params | 3 x 2048 x 512 = 3.146 M per expert; 32.2 B total |
| active expert params / token | 8 x 40 x 3.146 M = 1.007 B |
| non-expert matmul params / token | ~1.43 B (excluding embedding and output head); output head 0.51 B |
| prefill compute / token | ~4.9 GFLOP (2 x 2.44 B), excluding attention scores |
| attention compute / token at depth p | 163,840 x p FLOP (10 layers x 16 heads x 256 x 4 x p) |
| KV per token | 10,240 values: f16 20 KiB, q8_0 10.9 KB, kvarn5 about 6.9 KB |
| KV at 262,144 tokens | f16 5.4 GB, q8_0 2.85 GB, kvarn5 about 1.8 GB |
| experts at 3.5 / 4.25 bpw | 14.1 / 17.1 GB |

## The box: AUX-1

| item | value | source |
|---|---|---|
| GPU | RX 580 8 GB, gfx803, 36 CU, 2304 lanes, 1340 MHz | sysfs pp_dpm_sclk |
| GPU fp32 peak | 6.17 TFLOPS. fp16 is the same rate. No int8 dot product. No matrix cores. | ISA |
| GPU memory | 256 GB/s (2000 MHz GDDR5, 256-bit) | pp_dpm_mclk |
| BAR | 8 GB resizable BAR: all VRAM is CPU-mappable | dmesg |
| PCIe | 3.0 x16 (15.75 GB/s theoretical) | sysfs link speed/width |
| measured host-to-device | 6.82 pageable / 7.93 pinned GB/s (ggml-vulkan path) | llama-bells-selftest, 2026-10-15, DPM on |
| CPU | Ryzen 5 2600, 6 cores / 12 threads, Zen+, AVX2 on 128-bit units | lscpu |
| RAM | 32 GB DDR4-3000 dual channel (4 x 8 GB), 48 GB/s theoretical | dmidecode |
| second PCIe slot | chipset slot, about PCIe 2.0 x4 (about 2 GB/s) | board spec, lspci |

## Ceilings

### Prefill (compute-bound)

| case | FLOP / token | ceiling at 100% of peak | at 55% of peak |
|---|---|---|---|
| 1 card, zero context | 4.9 G | 1,260 t/s | ~690 t/s |
| 1 card, mean over a full 262k prompt, dense attention | 4.9 + 21.5 G | 234 t/s | ~130 t/s |
| 2 cards, zero context | 4.9 G | 2,520 t/s | ~1,380 t/s |
| 2 cards, 262k, attention cut to 10% by sparse prefill | 4.9 + 2.2 G | 1,740 t/s | ~950 t/s |

Consequences: 1000 t/s needs two cards. 500 t/s at 262k needs sparse prefill attention.
Expert streaming during prefill: the experts (14-17 GB) cross the bus once per ubatch.
At 7 GB/s that is 2-2.5 s per ubatch, so the ubatch must be >= 4096 tokens to stay under 1000 t/s.

### Decode (bandwidth-bound)

- All weights resident (impossible on one card, nearly possible on two): about 1.9 GB read per
  token at 3.5-4.25 bpw -> about 105 t/s at 200 GB/s effective.
- At 262k depth, dense attention adds about 1.8 GB of KV read (kvarn5) and 43 GFLOP per token.
  That alone is about 7 ms per token at peak. An MTP verify window multiplies the compute part.
  50 t/s at full depth needs sparse decode attention (or a very good MTP acceptance).
- On one card only about 4 GB of VRAM is left for experts (8 GB - 1.4 GB dense - 1.8 GB KV - buffers).
  That is about a quarter of the experts. The misses go to the CPU and the bus, in parallel:
  CPU (6 Zen+ cores, i-quants are compute-bound at about 2-4 GB/s per core) and PCIe (7-8 GB/s).

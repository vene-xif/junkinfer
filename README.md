# JunkInfer

Fast local inference of large mixture-of-experts models on junk-priced GPUs.
First target: **Qwen3.6-35B-A3B and its fine-tunes (`qwen35moe`) on the AMD RX 470/570/580 8 GB (gfx803, Polaris)**.

Status: early development. Nothing here is ready for use yet.

## Targets (dev box: Ryzen 5 2600, 32 GB DDR4-3000, RX 580 8 GB, PCIe 3.0 x16)

| | single RX 580 | dual RX 580 |
|---|---|---|
| decode, any depth up to 262k | >= 50 t/s | >= 80 t/s |
| prefill at zero context | as fast as physics allows (~600-700 t/s ceiling) | >= 1000 t/s |
| prefill at 262k context | as fast as physics allows | >= 500 t/s |
| weights quality | not below EXL3 3.5 bpw (measured KLD vs BF16) | same |
| KV-cache quality | not below q8_0 KV (measured KLD vs BF16) | same |

The release must run on one card. A second card is an optional speed-up.

## Layout

```
engineering-SIREN/   the engineering record (start at engineering-SIREN/index.md)
ref/                 the one-time BF16 reference run (VENE, RTX 3090) and its tools
LICENSE              Apache-2.0
NOTICES.md           third-party code and ideas this project uses
```

## Quality gates

Every change that touches numerics is measured as KL(BF16 || JunkInfer) against fixed
reference files made once from the BF16 model (`ref/`). See `engineering-SIREN/current/quality-gates.md`.

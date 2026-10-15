# Candidate ledger

> Every candidate considered, with its verdict and the reason. Read before proposing work.
> Append, do not rewrite. Nothing leaves this file.

Status: ACTIVE. Started 2026-10-15.

## From the 2026-10-15 survey (Strata, BELLS, ik_llama, BeeLlama, exllamav3)

| candidate | source | verdict | reason |
|---|---|---|---|
| Flat long-context speed | Strata | NOT TRANSFERABLE as is | It comes from the model (Qwen3.8-Flash-Next's trained sparse attention reads at most ~2048 positions), not from the engine. Qwen3.6 attention layers are dense. Training-free sparse attention is the only route (step 7). |
| GPU-hit / CPU-miss concurrent expert compute, doorbell in pinned memory | Strata | ACCEPTED (step 2) | Core of Strata's decode. Both halves busy at once; no driver sync on the critical path. |
| PCIe share of misses, share = L / (L + 1.85 R) | Strata | ACCEPTED (step 2) | Uses the copy engine beside CPU and GPU work. Tune on this box. |
| MTP verify windows (2.5-3.2 tokens/pass) | Strata | ACCEPTED (step 4) | Amortises the dense-weight read. Qwen3.6 ships an MTP layer. Note: more tokens per window = more distinct experts = more misses. |
| Route tail skip (skip missed experts of rank >= 7 in a window) | Strata | CANDIDATE | Lossy (KL ~0.003-0.006 on their model). Only if it passes the weights gate. |
| Adaptive cache seeded from a routing profile; slots sized in bytes | Strata | ACCEPTED | Routing traces come from the reference run. |
| Prefill in 8k chunks, next layer's experts streamed during current layer | Strata, BELLS | ACCEPTED (step 3) | The only way to reach high prefill when experts do not fit in VRAM. |
| KV streaming (only the read part of KV in VRAM) | Strata | DEPENDS on step 7 | Only useful with sparse attention. |
| BELLS expert cache (GPU computes every expert, misses copied) | BELLS | REVISED | On AUX-1 it measured 0.1-0.3 GB/s copy. Cause found: pageable mmap source -> ggml-vulkan `set_tensor_async` falls back to staging + a full queue drain per tensor (3 per expert). Also measured with DPM off. The design is fine; the copy path is broken on Vulkan. |
| BELLS prompt staging | BELLS | ACCEPTED as idea (step 3) | Needs a fast upload path on Vulkan (no CUDA D2D). |
| ggml-vulkan separate transfer queue | ggml | CHECK | Disabled for GCN by default; RADV on Polaris exposes no transfer-only family (graphics+compute family, 4x compute family). |
| Fused MoE up/gate | ik_llama | CANDIDATE | CPU and CUDA only in ik. Mainline has partial fusions. Port where the profile says it pays. |
| iqk CPU GEMV/GEMM kernels, row repacking | ik_llama | CANDIDATE (step 9) | The CPU computes the misses; Zen+ is compute-bound on i-quants. |
| IQ2/3/4_KT trellis quants | ik_llama | CANDIDATE (step 6) | QTIP family in GGUF; ik has a Vulkan IQ4_KT shader. KLD measured in the reference run. |
| KVarN KV cache | BeeLlama | ACCEPTED (step 5) | kvarn5/kvarn5 + 1024-token exact tail ~= q8_0 KLD at ~2/3 the size (their Qwen3.6-27B data). Vulkan shaders exist; untested on gfx803. |
| EXL3 weights | exllamav3 | CANDIDATE (step 6) | Pareto-best quality per bit. CUDA/ROCm only. Decode needs u32 multiply (quarter rate on gfx803) -> likely ALU-bound at ~half bandwidth. mul1 codebook byte-sum maps to `v_sad_u8`. Conversion needs a GPU: done once on VENE. |
| vLLM on gfx803 | prior work | REJECTED | Per-dispatch CP/qcm-fence stall at engine dispatch; all software axes exhausted (prior project). |
| HIP backend on AUX-1 | prior work | RISKY | Same ROCm stack as the vLLM stall. Vulkan is the default path. |

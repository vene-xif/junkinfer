# JunkInfer Codebase Navigation Map

## Where JDE Hooks into llama.cpp

| Call Site | Function |
|-----------|----------|
| `src/llama-context.cpp:599` | `jde_init()` — initialize JDE engine (called at context creation) |
| `src/llama-context.cpp:601` | `jde_kv_attach()` — attach JDE to KV memory and install I/O hooks |
| `src/llama-context.cpp:2124` | `jde_kv_guard()` — guard elastic expert slots from context writes |
| `src/llama-context.cpp:2129` | `jde_prefill_can()` — check if prompt ubatch can use JDE |
| `src/llama-context.cpp:2142` | `jde_prefill()` — run prefill on JDE instead of graph path |
| `src/llama-context.cpp:2155` | `jde_can_decode()` — check if single-token ubatch can use JDE |
| `src/llama-context.cpp:2188` | `jde_kv_written()` — mark KV cells written, invalidate sparse attention metadata |
| `src/llama-context.cpp:3870,3922` | `jde_state_sync()` — sync TP2 helper GPU state before save/load |
| `src/llama-context.cpp:3939` | `jde_state_partial()` — recurrent state only after checkpoint load |
| `src/llama-kv-cache.cpp:1284-1305` | `llama_kv_hm_get_hook / llama_kv_hm_set_hook` — KVarN-6 head-major state I/O |
| `src/llama.cpp` (metadata mode) | `exl3_open()` — load exllamav3 checkpoint directory (native, no GGUF conversion) |
| `src/llama-exl3.h` | EXL3 loader declarations (transforms, tensor producer) |
| `src/llama-exl3.cpp` | EXL3 loader: safetensors reader, gguf_context builder, T1-T6 transforms |
| `src/junkinfer/jde_signal.cpp:31` | `jde_signal_install()` — install signal handlers to release TP GPU on exit |

## jde.cpp Sections (5365 lines)

| Lines | Section | Purpose |
|-------|---------|---------|
| 48-140 | Namespaces & structs | `vbuf`, `bind`, `pipe`, `pool`, `jde_engine` (main engine state, buffers, GPU resources) |
| 140-995 | `jde_engine` struct | Engine fields: buffers (vram_bufs, stg_*), GPU queues, command buffers, profiling, TP2 state, MTP head |
| 996-1025 | Forward declarations | Helper functions, KVR declarations, TP declarations, MTP declarations |
| 1026-1049 | Checks & guards | `jde_is_x3()`, `jde_wtype_of()`, `jde_supported_tensor()` |
| 1050-1133 | EXL3 setup | `jde_x3_init()` — initialize EXL3 codebook and operations |
| 1134-1422 | Engine init | `jde_init_stage()` — allocate buffers, load shaders, init weights; `jde_layer_backend()` |
| 1423-1480 | Layer backend dispatch | Decide which GPU handles each layer (single or split) |
| 1433-1480 | TP selftest | `jde_tp_selftest()` — validate TP2 sync on startup |
| 1481-1608 | TP2 setup | `jde_import_host()`, `jde_tp_wait()`, `jde_tp_mark()`, `jde_tp_record_helper()`, `jde_tp2_helper_alloc()`, `jde_tp_signals()` |
| 1609-1696 | TP2 relay | `jde_tp_relay()` — host CPU loop feeding the helper GPU during TP decode |
| 1697-1989 | TP2 init full | `jde_tp_init()` — full TP2 setup (queues, semaphores, events, relay thread) |
| 1990-2023 | TP2 cleanup | `jde_tp_free()` — release TP2 resources |
| 2024-2100 | TP2 full MoE | `jde_tp2_reduce()`, `jde_tp2_gdn()`, `jde_tp2_ffn()`, `jde_tp2_moe_reduce()`, `jde_tp2_router_helper()` |
| 2100-2312 | TP2 state copy | `jde_tp2_state_regions()`, `jde_tp2_state_copy()`, `jde_tp2_state_in/out()` |
| 2314-2512 | TP2 state record | `jde_sreg`, `jde_tp2_kv_copy()`, `jde_tp2_meta_rebuild()`, `jde_tp2_mirror()` |
| 2514-2941 | MTP sidecar | `jde_mtp_release()`, `jde_mtp_upload()`, `jde_mtp_can_decode()`, `jde_mtp_chunk()`, `jde_mtp_decode()` — draft head on TP2 helper |
| 2984-3029 | Profiling & tests | `jde_bench()` — throughput measurement |
| 3030-3290 | Attention tests | `jde_attn_test_kvr()`, `jde_attn_test_kvr2()`, `jde_attn_test()` — unit tests for sparse attention variants |
| 3291-3344 | Cleanup & utils | `jde_free()`, `jde_hybrid()` |
| 3357-3404 | Decode gate | `jde_can_decode()`, `jde_zero_rs()` — check ubatch fit, zero recurrent state |
| 3405-3575 | Elastic slots KV | `jde_el_init()` — expert cache slots carved from KV cache tail |
| 3497-3576 | KV hooks | `jde_note_kv_clear()`, `jde_kv_attach()`, `jde_kv_guard()` — manage elastic slots on KV writes |
| 3577-3743 | Elastic slots compute | `jde_cb_evict_below()`, `jde_cb_evict_all()`, `jde_compute_buffer()`, `jde_compute_evict()` — expert cache on compute buffer |
| 3617-3707 | State I/O hooks | `jde_kv_written()`, `jde_tensor_get_batch()`, `jde_state_sync()`, `jde_state_partial()`, `jde_kv_invalidate()` |
| 3743-3999 | EXL3 matvec | `x3_pos_rc()`, `x3_mcg()`, `x3_mul1()`, `x3_mul()`, `x3_scale()`, `jde_x3_dequant()` — EXL3 trellis codebook ops |
| 4000-4132 | Async prefill fill | `jde_fill_poll()` — wait for / publish asynchronous expert-cache fills (own queue) |
| 4133-4232 | CPU experts | `jde_cpu_experts()` — route selected experts to CPU (EXL3, MoE misses) |
| 4233-4767 | Sparse attention | `jde_attn_core()` — the main decode attention kernel, handles dense + sparse (topk, HM, KVarN-6, background) |
| 4450-4700 | Page selection | Sparse page scoring, selection, background estimation ops |
| 4768-5150 | Attention MoE dispatch | `jde_dispatch()` — route tokens through attention, sparse logic, MoE handler dispatch |
| 5150-5280 | Decode stage impl | `jde_decode_stage()` — single TP stage: attention + FFN + MoE + RMS + GDN + gating |
| 5281-5365 | Public decode | `jde_decode()`, `jde_can_decode()` — main entry point; cascades through TP stages |

## jde_prefill.inc Sections (1361 lines)

| Lines | Section | Purpose |
|-------|---------|---------|
| 1-17 | Header | JDE prefill design docs reference |
| 18-24 | Config checks | `jde_pf_tp()`, `jde_pf_on()` — check tensor-parallel and enable flags |
| 25-46 | Config structs | `jde_pf_dvar`, `jde_pf_iv()` — prefill variant selection, input format |
| 47-126 | GEMM dispatches | `jde_pf_gemm()`, `jde_pf_gemm_w()`, `jde_pf_gemm_id()`, `jde_pf_prep_id()` — dense/expert GEMMs, variants |
| 127-250 | Submission/warmup | `jde_pf_ms_init()`, `jde_pf_submitv()`, `jde_pf_need_r()`, `jde_pf_nmb()`, `jde_pf_warm()` |
| 251-312 | Config parse | `jde_set_spf_dense_from()` — sparse prefill density threshold |
| 313-380 | Rank dispatch | `jde_spf_dense_from`, `jde_pf_rank()` — per-head segment dispatch logic, sparse vs dense |
| 381-1035 | Rank execute | Segment execution (main loop over per-position computations for each head/position) |
| 1036-1044 | Completion | `jde_pf_wait_done()` — wait for all segments |
| 1046-1361 | Main loop | `jde_prefill()` — top-level prefill path, ubatch → multi-GPU segments, state/logits output |

## jde_kvr.inc Sections (351 lines)

| Lines | Section | Purpose |
|-------|---------|---------|
| 17-25 | Enabled & hooks | `jde_kvr_enabled()`, `jde_kvr_hm_get/set()` declarations — KVarN-6 feature gate, I/O hooks |
| 26-26 | Global state | `g_kvr` — KVarN engine singleton |
| 27-45 | Alloc | `jde_kvr_alloc()` — allocate sealed-record buffer and staging |
| 47-95 | Init | `jde_kvr_init()` — enable hooks, initialize records buffer from KV cache |
| 96-151 | Seal/unseal | `jde_kvr_head1_back()`, `jde_kvr_seal()`, `jde_kvr_unseal()` — tile sealing (q8_0 → 6-bit codes + axes) |
| 152-192 | Prefill materialize | `jde_kvr_pf_materialize()`, `jde_kvr_pf_copyout()` — output sealed tiles during prefill, materialized rows |
| 194-255 | I/O hooks impl | `jde_kvr_which()`, `jde_kvr_buf()`, `jde_kvr_hm_get()`, `jde_kvr_hm_set()` — q8_0 row pack/unpack for head-major state |
| 316-351 | Metadata rebuild | `jde_kvr_meta_rebuild()` — after state load, rebuild sparse attention page statistics |

## Shaders (60 .comp files)

### Decode Attention
| Name | Purpose |
|------|---------|
| `attn_decode.comp` | Single-token attention, q8_0 K/V, one workgroup per head |
| `attn_gqa.comp` | Split-K GQA, n_kv ≤ 8192, f16 K/V |
| `attn_long.comp` | Split-K GQA for long contexts (n_kv > 8192), f16 K/V |
| `attn_d2.comp` | Split-K GQA for long contexts (n_kv > 8192), q8_0 K/V |
| `attn_hm.comp` | Split-K GQA for long contexts over head-major q8_0 K/V (llama_kv_cache) |
| `attn_hmr.comp` | Split-K GQA for long contexts with KVarN-6 (page-based, head-major records) |
| `attn_kvr6.comp` | KVarN-6 sealed records + exact window, sparse/dense |
| `attn_kvr6q.comp` | KVarN-6 pre-pass: q·Bk per tile, query head, K slice |
| `attn_kvr.comp` | KVarN-6 attention over 6-bit codes + dual-axis normalization |
| `attn_merge.comp` | Split-K combine: softmax-weighted sum of partials |

### Sparse Attention
| Name | Purpose |
|------|---------|
| `attn_qk.comp` | Prep: q (rms + RoPE) → qo, k (rms + RoPE) → cache |
| `attn_score.comp` | Page score: upper bound q·k per page |
| `attn_topk.comp` | Page selection: top-K + sink/recent, radix sort |
| `attn_bg.comp` | Tail estimation: background pages (mode 1 or 2) |
| `attn_bgx.comp` | Diagnostic variant: split error into page mass / value components |
| `attn_meta.comp` | Page statistics: mean K, radius, variance, mean V, cell count |
| `attn_kvsim.comp` | KVarN quality test: q8_0 tiles → KVarN → reconstruct |

### Matvec (Decode)
| Name | Purpose |
|------|---------|
| `mv.comp` | y = W x for NT tokens, f32/f16 native format |
| `x3_mv.comp` | EXL3 mat-vec raw part, 6-bit codes + axes, multiple BITS (2-8) and variants (FUSEP, FUSEQ, GROUP) |
| `x3_mvp.comp` | FUSEP variant: input prep fused into mv |
| `x3_mvq.comp` | FUSEQ variant: output quant fused into last wave |
| `x3_post.comp` | Output stage: KS reduction, Hadamard transform, scale by svh |
| `x3_prep.comp` | Input stage: Hadamard, input transform (plain / rms / silu), scale by suh |
| `moe_wsum.comp` | MoE combine: sum expert outputs with weights |
| `combine.comp` | Shared expert gate + routed experts + attention gate |

### Prefill (JDE)
| Name | Purpose |
|------|---------|
| `pf_mm.comp` | Prefill GEMM (EXL3 trellis weights), dense and expert variants, windowed for TP |
| `pf_mv_id.comp` | Prefill narrow expert tile mat-vec (remainders <= 8 rows): register-based, 4 stripes per pass |
| `pf_qk.comp` | Prefill attention prep: q/k rms + RoPE → cache, T positions |
| `pf_fa.comp` | Prefill flash attention, 4 positions per workgroup, q8_0 K/V, sparse/dense |
| `pf_fa_sparse.comp` | Sparse prefill attention: page-based sampling + background |
| `pf_pmeta.comp` | Sparse prefill: page statistics for tail estimation |
| `pf_attn_out.comp` | Prefill attention output: weighted V sum (variants with q8_0) |
| `pf_rms.comp` | Prefill RMS normalize |
| `pf_silu_mul.comp` | Prefill SiLU gate |
| `pf_f32mv.comp` | Prefill f32 matvec |
| `pf_mm.comp` | Prefill quantized matvec (EXL3) |
| `pf_copy.comp` | Prefill data copy |
| `pf_copyr.comp` | Prefill routed copy |
| `pf_combine.comp` | Prefill combine (shared + routed) |
| `pf_add.comp`, `pf_add2*.comp` | Prefill add (residual, format variants) |
| `pf_topk.comp` | Prefill top-K expert selection |
| `pf_rowmap.comp` | Prefill row mapping (tile dispatch for dense, wide experts, narrow expert remainders) |
| `pf_prep.comp` | Prefill input prep |
| `pf_gdn_*.comp` | Prefill GDN conv, rule, state |

### Foundation / FFN
| Name | Purpose |
|------|---------|
| `rmsnorm.comp` | RMS normalize: y = x / sqrt(mean(x²) + eps) * w |
| `silu_mul.comp` | SiLU gating: y = silu(x) * w |
| `gate_mul.comp` | Attention output gate: out *= sigmoid(gate) |
| `router.comp` | MoE router: softmax over logits, top-K selection |
| `gdn_conv.comp` | GDN conv1d kernel 4 + SiLU, state shift |
| `gdn_fused.comp` | GDN fused for ≤4 tokens: conv + norm + gated delta |
| `gdn_step.comp` | GDN single-step gated delta rule |

### KVarN-6
| Name | Purpose |
|------|---------|
| `kvr_seal.comp` | KVarN seal: q8_0 tile → 6-bit codes + dual-axis axes, per tile |
| `kvr_mat.comp` | KVarN materialization: sealed tiles → q8_0 rows (prefill output, state save) |

### Tensor Parallel
| Name | Purpose |
|------|---------|
| `tp_split.comp` | TP2: split routed selections between GPUs |
| `tp_add.comp` | TP2: all-reduce final step (x += p0 + p1) |

## Environment knobs (the ones that matter)

Every knob is read with `getenv("JI_...")` in `src/junkinfer/*.cpp|*.inc`; the comment next to each read is the
authoritative description: `grep -n 'getenv("JI_' src/junkinfer/*` lists them all (~140, most are experiments or
diagnostics). The production values are set by `scripts/serve.sh` / `serve_prod.sh` / `serve_boot.sh`.

| knob | default | meaning |
|---|---|---|
| `JI_DECODE` | serve.sh sets 1 | JDE decode on (else llama.cpp's graph decode) |
| `JI_JPF` | serve_prod / boot: 1 | JDE prefill (else the ggml graph prefill) |
| `JI_TP` | 0 (prod 2) | two-GPU mode: 2 = tensor-parallel decode (+ two-card prefill with JI_JPF), 1 = routed experts only on GPU 2 |
| `JI_TP_MARGIN_MIB` | prod 1500 (1900 with MTP) | VRAM the helper GPU keeps free (prefill scratch, MTP head) |
| `JI_TP_HOT`, `JI_TP_FRAC`, `JI_TP_DUP`, `JI_TP_SLOTS` | ref/tp/tp_hot_*.bin, 0.7, 0.25, auto | TP2 expert ownership: routing-count file, helper's share of traffic, share held by both cards, slot cap |
| `JI_MTP` (serve.sh) | prod 1 | MTP speculative decoding (sidecar draft head on the helper); `JI_MTP_PMIN` 0.5, `JI_MTP_VOCAB_FILE`, `JI_MTP_HEAD_TYPE` iq4_nl |
| `JI_CACHE_MIB` | auto | VRAM expert cache size (auto = free VRAM minus `JI_CACHE_MARGIN_MIB`, 400 for EXL3) |
| `JI_CACHE_SWAPS` | serve.sh 16 | expert-cache evictions per token |
| `JI_ELASTIC`, `JI_EL_CB` | 1, 1 | elastic expert slots in the free KV tail / in the idle compute buffer |
| `JI_STATIC` | 1 (0 under JI_PROFILE_GPU) | replay recorded per-token command buffers |
| `JI_SPARSE` | 65536 | sparse decode floor budget in keys (0 = dense) |
| `JI_SPARSE_FRAC` | 0.5 | decode budget = max(JI_SPARSE, FRAC x context) |
| `JI_SP_SEL` | 1 | rank pages by estimated attention mass (0 = Quest-style bound) |
| `JI_SP_MIN` | 32768 | sparse decode from this many cells |
| `JI_SP_BG` | 2 | tail estimate of unselected pages (1 means only, 2 + key variance; 8/9 diag) |
| `JI_SP_RAW` | 1 | page statistics in the raw (un-rotated) key basis |
| `JI_SP_DENSE_LAYERS` | none | e.g. 3,7,11: these layers stay dense (quality setting) |
| `JI_SP_ORACLE` | 0 | diag: rank pages by exact mass |
| `JI_SPARSE_PF` | 32768 | sparse prefill floor budget in keys (0 = dense prefill) |
| `JI_SPARSE_PF_FRAC` | 0.5 | prefill budget = max(JI_SPARSE_PF, FRAC x keys) |
| `JI_SPF_SEL` | 1 | prefill ranking by estimated log-mass |
| `JI_SPF_LOCAL`, `JI_SPF_DENSE_TAIL` | 4096, 4096 | prefill local window; the prompt's last N positions dense |
| `JI_CKPT_SPLIT_MIN` | 2048 | llama-server: min tokens after the last checkpoint before it allows a split there (0 = old behavior: split at every context checkpoint, e.g. at user message start in chat) |
| `JI_KVARN` | 0 | KVarN-6 KV cache (opt-in; see features/kvarn6-kv-cache.md) |
| `JI_KVARN_RING` | 12544 | KVarN exact window ring, cells per head |
| `JI_KVR_TAIL` | 1 | put the rings in the KV tensors' unused tail (0 = own buffers, costs 130 MiB VRAM) |
| `JI_KVR_ITERS` | 16 | KVarN seal: balancing iterations per tile |
| `JI_KVR_PFT` | 0 | test: force the separate prefill copy buffer |
| `JI_KVSIM` | 0 | diag: simulate KVarN at N bits inside the q8_0 cache |
| `JI_PROFILE` | 0 | every N tokens: per-token phases, expert hits / misses per step size |
| `JI_PROFILE_GPU` | 0 | per-kernel GPU time (timestamps after every dispatch) |
| `JI_TP_PROF`, `JI_TP_GPROF` | 0 | TP2 per-layer timestamps / helper per-kernel time |
| `JI_PF_GPROF`, `JI_PF_TL` | 0 | prefill per-kernel time / segment timeline |
| `JI_PF_XQ` | 1 | TP prefill exchange on its own queue (0 = copies inline; S10 soak) |
| `JI_PF_UDMA` | 0 | **do not use** (hard-locked the box 2026-10-22) |
| `JI_ATTN_TEST`, `JI_ATTN_KVR2`, `JI_TP_TEST` | 0 | kernel self-tests (diag) |
| `JI_KV_DUMP*`, `JI_ROUTE_DUMP` | unset | debug dumps |
| `JI_X3_FOLD` | 2 | mul1 decode variant: 0 exact (bit-identical), 1 centred byte sum, 2 raw affine fold (x3_mv decode only) |
| `JI_EXL3_CHECK` | unset | debug: load a checkpoint directory and oracle GGUF in parallel, compare every tensor (bytes and metadata keys), exit with results |
| `JI_EXL3_DUMP` | unset | debug: load checkpoint directory; dump named tensors after transforms and re-quantization (e.g., `eh_proj`, `attn_v`) to stderr |
| Model path syntax | — | `<path>` = GGUF file or exllamav3 checkpoint directory (config.json + model.safetensors.index.json + quantization_config.json); `<dir>::mtp` = checkpoint's MTP layer loaded and re-quantized at load (IQ4_NL dense, Q5_K attn_v, Q6_K head) |
| `JI_HOST_THP` | 1 | imported host buffers on 2 MiB huge pages with pre-faulting (0 = old 4 KiB pages, ~8-25 % slower) |
| `JI_PF_NARROW` | 1 | use narrow expert tile kernels (pf_mv_id.comp) for routed-expert remainders <= 8 rows (0 = old pf_mm_id32 path) |
| `JI_PF_KEEPCB` | 0 | keep prefill command buffers between prefills (1 = no pool trim, slower for varied ubatch sizes) |

## Project Scripts

### scripts/
| File | Purpose |
|------|---------|
| `serve.sh` | Single-GPU server (MVP, Qwen chat, Vulkan JDE) |
| `serve_boot.sh` | Boot-time server (auto-detects GPU count, adapts config) |
| `serve_prod.sh` | Production server (two RX 580s, TP2 decode, TP prefill, LAN) |
| `smoke_decode.sh` | Short smoke test (1k prompt + 128 tokens) |
| `depth_kv.sh` | Depth test from saved KV state (prefill once, decode at depth) |
| `depth_bench.py` | Depth throughput benchmark generator |
| `api_suite.py` | API test suite (OpenAI/Anthropic format) |
| `smoke_api.py` | Short API smoke test |
| `cpu_performance.sh` | CPU governor control (perf mode, systemd unit) |
| `thermal_guard.sh` | GPU temperature monitor (systemd, throttle on overheat) |
| `big_bench.py` | Large prompt builder (source code corpus ~116k tokens) |
| `junkinfer.service` | Systemd unit for server |
| `ji-thermal-guard.service` | Systemd unit for thermal logger |

### tools/
| Dir/File | Purpose |
|----------|---------|
| `cachesim.py` | Expert cache simulator |
| `swap_tensors.py` | Tensor swap utility |
| `swap_head.py` | Head swap utility |
| `mtp_extract.py` | MTP verification data extraction |
| `exl3/` | EXL3 tools: `exl3_to_gguf.py`, `exl3_ref.py` (reference implementation) |
| `aux-jt/` | AUX-1 test tools: profiling (`srvpf.py`, `verprof.py`), coverage (`vocabcov.py`), smoke (`srtest.sh`, `srcross.sh`), variants (`varhm.sh`, `varf.sh`, `vard2.sh`), sparse gating (`spfgate.sh`), expert probe (`x3probe.sh`), etc. |

### ref/
| Dir/File | Purpose |
|----------|---------|
| `run_ref.sh` | Reference run script (full setup, measurements) |
| `eval_cpu.sh` | CPU expert evaluation (direct compute) |
| `prompts/` | Test prompts |
| `mtp/` | MTP verification data |
| `tp/` | TP validation |
| `tools/` | Reference utilities |

### lab/
| Dir/File | Purpose |
|----------|---------|
| `mmbench/` | Prefill routed-expert GEMM micro-benchmark: mmbench.cpp + build.sh, synthetic EXL3 experts with skewed routing, kernel variant testing, VGPR inspection |
| `matvec/` | Matvec kernel lab / tuning |
| `xsync/` | Cross-GPU sync lab / experiments |
| `xferbench/` | Transfer benchmark (PCIe throughput) |

## Build (Shader Compilation & Link)

**Shader Compilation:** CMake (Vulkan-only, opt-in with `-DGGML_VULKAN=ON`)

- **Compiler:** `glslc` (Khronos, `--target-env=vulkan1.2 -O`)
- **Shader source:** `src/junkinfer/shaders/*.comp`
- **Output:** `${CMAKE_BINARY_DIR}/bin/jde-shaders/*.spv`
- **Rule:** Per-shader `.comp` → `.spv` (Vulkan SPIR-V bytecode)
- **Variants:** Multi-specialization (constants `-D` flags):
  - `mv.comp`: types (0-4) × tokens per pass (1-4) = 20 variants + MOE variants
  - `x3_mv.comp`: bits (2-8) × tokens (1-4) × (FUSEQ / FUSEP / GROUP) = 100+ variants
  - `pf_mm.comp`: dense (BN64/BN128, BK16) and ID variants
  - `pf_fa.comp`: positions (P=4, 8, 16), page granularity (SUB=64/128)
  - Attention (attn_qk, etc.): q8_0 KV variants

**Build Command (AUX-1, Vulkan, 12 parallel jobs):**
```bash
cd /path/to/engine
cmake --build build -j12
```

**Output:** Engine binary `build/bin/llama-server` (includes all shaders as resources)

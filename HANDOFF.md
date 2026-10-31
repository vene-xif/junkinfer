# JunkInfer — handoff

Written 2026-10-26 by the agent that did the work from 2026-10-15 to 2026-10-26, for the next engineer or LLM agent.
The project is **paused** at this point. The box serves the model on its own (boot service). Read this file
first, then `engineering-SIREN/index.md` and `engineering-SIREN/current/STATE.md`.

## 1. What this is

**Served model since 2026-10-26: Sage-A1.5** (InternScience; a fine-tune of the same architecture, EXL3 4.0 bpw from
`jyohukuchan/Sage-A1.5-exl3-4.0bpw`, GGUF `/mnt/ssd/models/Sage-A1.5-exl3-4.00bpw.gguf`, alias `sage-a1.5`).
Qwen3.6 stays available (`JI_MODEL=/mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf` in `/etc/junkinfer.env`).
All numbers below are for Qwen3.6 unless marked Sage; Sage notes are in §2.

Fast local inference of **Qwen3.6-35B-A3B** (hybrid MoE: 40 layers, 10 full-attention + 30 Gated DeltaNet,
256 experts, 8 active, A3B) on **AMD RX 580 8 GB** cards (gfx803 / Polaris, Vulkan via RADV). Weights: community
**EXL3 4.0 bpw**, converted to GGUF (`tools/exl3/exl3_to_gguf.py`). The engine is a **llama.cpp fork** with a
custom Vulkan decode/prefill engine, **JDE** (`engine/src/junkinfer/`). llama-server provides the API.

Targets (README): single card decode >= 50 t/s at any depth; two cards decode >= 80 t/s, prefill >= 1000 t/s at
zero context and >= 500 t/s at 262k; quality not below EXL3 3.5 bpw weights and q8_0 KV.

## 2. Current state (2026-10-31)

**Production = boot service on AUX-1** (`junkinfer.service` -> `scripts/serve_boot.sh` -> `scripts/serve.sh`):

- `http://192.168.1.4:8080/v1` (OpenAI: chat / completions / responses; Anthropic: `/v1/messages`), web UI at `/`.
  No API key (set `JI_API_KEY` in `/etc/junkinfer.env`). Model alias `sage-a1.5`.
- Two cards enumerated: TP2 tensor-parallel decode + two-card JDE prefill + MTP speculative decoding
  (= `scripts/serve_prod.sh` + native EXL3 + Sage's own MTP head). One card: single-card JDE. q8_0 KV, 262,144 context,
  sparse attention on, thinking on, **Qwen-Sharp chat template** (`ref/templates/qwen_sharp.jinja`).
- Restarts on exit. Verified across two reboots on 2026-10-31: server, thermal guard, fan curve, governor all up.
  `scripts/smoke_api.py`: ALL PASS (chat, tool-call parsing, prefix cache 6839 -> 23 tokens, prefill 537 t/s).
- **Short-prompt prefill fixed (engine 14eb6fe, f519a49):** llama-server was splitting every chat prompt at the start of the last user message,
  running two-batch prefills for short prompts (~1-2 s overhead). New `JI_CKPT_SPLIT_MIN` (default 2048) prevents checkpoint
  splits unless at least that many tokens lie after the checkpoint. Multi-turn cache reuse unchanged (verified: depth-kv tests ALL PASS).

**Sage-A1.5 (boot default, 2026-10-31):** Native EXL3 checkpoint directory `/mnt/ssd/models/Sage-A1.5-exl3-4.0bpw`
(no GGUF conversion). TP2, **own MTP head** (p-min 0.8), Qwen-Sharp template, alias `sage-a1.5`. Decode rewrite / story /
explain 47.4 / 62.8 / 55.3 t/s; prefill 537 t/s (6.8k, batch 1); smoke_api ALL PASS. MTP acceptance 87 / 95 / 91 %. Load ~250 s
(native 143 s + MTP re-quant 107 s). vs 2026-10-26 (39.6 / 45.0 / 41.7, no MTP): **+22 / +41 / +36 %**. Sage's own MTP
head (IQ4_NL, Q5_K, Q6_K): outputs identical at every p-min (lossless); Qwen3.6 sidecar on Sage gives no gain (64/88/85 %).

**Measured speed** (AUX-1, both GPUs capped at 100 W, EXL3 4.0 bpw, q8_0 KV, sparse defaults, >= 1024 tokens):

| | value | config / date |
|---|---|---|
| decode, short context | 51-61 t/s (rewrite / story / explain 51.6 / 61.1 / 55.8) | TP2 + MTP, 2026-10-25 |
| decode, short, no MTP | ~59 t/s | TP2, 2026-10-21 |
| decode at 200k | 32.4 t/s (dense 25.6) | TP2, no MTP, 2026-10-24 |
| prefill, ~7k prompt | ~530-555 t/s | TP2 JDE prefill, 2026-10-23 |
| prefill, 151k prompt | 288-293 t/s (dense 224) | TP2, sparse prefill, 2026-10-24 |
| single card | decode ~46 short / ~34 at 100k / ~30 at 245k | 2026-10-21 (before the sparse re-tune) |

**Quality** (KL vs our own full-attention output, v2 references, code text, last 1024 positions): sparse decode
0.0001 (64k) .. 0.0011 (261k); sparse prefill 0.0009 (130k) / 0.0014 (261k); end to end 0.00125 (130k) / 0.00165
(261k). For scale, q8_0 KV vs f16 KV costs ~0.001-0.0015. Weights vs BF16: see `quality-gates.md`.

**Gap to targets:** decode short ~55 vs 80; prefill ~550 vs 1000. Prefill is power-bound at 100 W
(throughput ~ watts); the 145 W runs wait for a new PSU (see §6).

## 3. Machines and access

| name | role | access |
|---|---|---|
| AUX-1 | the target box: Ryzen 5 2600, 32 GB DDR4, 2x RX 580 8 GB, B450, Ubuntu | `ssh vene@192.168.1.4` (key auth). `sudo` needs the user's password: **ask the user**; never write it into files |
| VENE | canonical git repos, BF16 reference data. RTX 3090 is **off-limits** (the user uses it) | `vene@192.168.1.2`, repos in `/mnt/chonker/projects/junkinfer` (+ `/engine`) |
| CHERRY | the agent's workstation (scratchpad clones; volatile `/tmp`). Its GPU is off-limits | 192.168.1.3 |

AUX-1 GPUs: **card1 / Vulkan1 = PCIe 3.0 x16 card = primary** (attention, experts streamed from host);
**card0 / Vulkan0 = chipset PCIe 2.0 x4 card = TP helper** (hot experts resident, MTP head). If only one card
enumerates, it becomes Vulkan0 (`JI_GPU=0`; serve_boot.sh handles it).

AUX-1 paths: models `/mnt/ssd/models/` (run models only from the SSD; **Sage-A1.5 native checkpoint:**
`/mnt/ssd/models/Sage-A1.5-exl3-4.0bpw/`), SSD overflow (fill HDD) `~/models/ssd-overflow/` (MiniCPM-V 4.7 GGUF moved
there 2026-10-31), saved KV / states `/mnt/ssd/kvcache/` (KL prefix states `jref_v2/`), harness `~/jt/` (snapshot
in `tools/aux-jt/`), engine `~/junkinfer/engine` (rsynced, **not a git repo**), project `~/junkinfer` (scripts, ref, logs).

## 4. Rules the user set (follow them)

- **One model process at a time** on AUX-1 (pinned experts: two processes OOM the box). The boot service holds the
  GPUs: `sudo systemctl stop junkinfer` before any test, `start` afterwards.
- **Never SIGKILL a busy two-card (TP2) process.** Ending one mid-step leaves the other GPU waiting on an event; the
  Polaris reset fails. SIGTERM is handled (engine releases its events), but don't kill during a step if avoidable.
- Downloads and `pip install` need the user's permission. (The full API suite needs `pip install openai anthropic`;
  not installed on AUX-1 or CHERRY as of 2026-10-26.)
- Keep the 100 W GPU power cap (`/etc/udev/rules.d/31-amdgpu-powercap.rules`) until the user swaps the PSU.
- Don't use `JI_PF_UDMA` (hard-locked the box).
- VENE: CPU-only jobs, `nice 19`, <= 4 threads; never touch its beellama on port 8100. Never use CHERRY's GPU.
- Kill processes only by PID or exact-args match.
- **Every decode speed test generates >= 1000 tokens** (ignore_eos); the expert cache needs warm-up.
- **Report tg and prefill t/s before -> after for every improvement** (and KL for anything touching numerics).
- Update the docs with every improving commit (STATE.md in place; engineering log appended; feature docs).
- Use cheap sub-agents (Haiku) for simple, well-specified work; keep AUX-1 GPU runs under direct control.
- Long experiments get an active watcher (poll the log / dmesg / reachability), so a crash or hang is noticed.
- Docs style: SIREN (`engineering-SIREN/SIREN.md`) — short sentences, dated entries, numbers with build id.

## 5. Workflow

1. Clone both repos from VENE into a scratch dir (`git clone vene@192.168.1.2:/mnt/chonker/projects/junkinfer`,
   and `.../junkinfer/engine`, branch `junkinfer`). VENE repos accept pushes to the checked-out branch.
   Commit identity used so far: `vene <kszmit24@gmail.com>`, with a `Co-Authored-By:` line for the agent.
2. Edit, commit, push to VENE.
3. Sync the engine to AUX-1 from **one** source dir: `rsync -a engine/src/junkinfer/ vene@192.168.1.4:junkinfer/engine/src/junkinfer/`
   (multi-source rsync flattens into the target root). Scripts: `rsync -a scripts/ vene@192.168.1.4:junkinfer/scripts/`.
4. Build on AUX-1: `cd ~/junkinfer/engine && cmake --build build -j12` (shaders compile to SPIR-V in the build).
   Stop the service first if the build replaces the running binary's shaders.
5. Measure (§7), document, commit.

If VENE is unreachable: both repos also have an `aux` remote (bare repos on AUX-1 `~/git-backup/`).

## 6. Box operations and hazards

- **Boot services** (all enabled): `junkinfer` (server), `ji-thermal-guard` (logs `~/junkinfer/logs/thermal.log`,
  previous boot in `thermal.prev.log`; SIGTERM to the model at 88 C), `ji-fancurve` (aggressive fan curve,
  `/usr/local/bin/ji-fancurve.sh`), `junkinfer-cpu-performance` (governor `performance`; schedutil costs 6-25 %
  decode). Boot units must not use `After=multi-user.target` (ordering cycle; systemd drops them).
- Kernel params: GTT 24 GB (`ttm.pages_limit=6291456 amdgpu.gttsize=24576`; pinned experts need it),
  `iommu=pt`, `amdgpu.lockup_timeout=60000,120000,60000,60000`, `amdgpu.gpu_recovery=0` (a GPU hang leaves the box
  up; the hung card then needs a **cold power cycle** by the user — a warm reboot does not bring it back). DPM
  forced high by udev.
- **Power delivery is suspect.** The box powered off without any log entry twice under heavy two-card load
  (2026-10-24 and 2026-10-25 01:47, the latter during a 151k two-card prefill; GPUs at 66 / 63 C, so not
  thermal). The x4 card fell off the PCIe bus twice at 145 W (2026-10-21). The mSATA SSD adapter logs SATA link
  resets independent of GPU load. GPU power cap 100 W permanent (user decision 2026-10-30).
- **TP2 prefill ring hang** (open-paths S10): ~2 hangs in ~10 h of two-card prefill. Check `dmesg` for
  "ring ... timeout". Recovery: reboot; if a card is missing afterwards, cold power cycle.
- Drop the page cache (`echo 3 > /proc/sys/vm/drop_caches`) before speed runs after big file reads (~10 % at 245k).

- **Conversation switches** (two bugs fixed 2026-10-26, engine 5b3e708: KV-tail slot zeroing on restore, stale MTP rows): run
  `scripts/conv_switch_test.py` after any change to state save/restore, the KV guard / elastic slots or MTP.

## 7. How to measure

- **KL** (`tools/aux-jt/kl_v2.sh`, on AUX-1 `~/jt/kl_v2.sh SET ROW ENV...`): compares against **v2** full-attention
  references (`~/jt/*_v2.jref`, prefix states `/mnt/ssd/kvcache/jref_v2`). Decode-only by default (prefix
  restored); `JR_STATE_DIR=<empty dir>` prefills from scratch. Rows: L_code 0/1/2 = 32k/130k/261k, L_code64,
  L_code98. Two cards: `JI_TP=2`. Single card: kl_v2.sh defaults to `GGML_VK_VISIBLE_DEVICES=1` (the x16 card);
  with only one card present use `0`. (The 2026-10-24/25 "single card" KL rows ran with `JI_TP=0
  GGML_VK_VISIBLE_DEVICES=0`, i.e. on the x4 card: valid for KL, not for speed.)
  **Never build references on the ggml graph path with large Vulkan blocks**: it loses layer-3 KV writes beyond
  ~8k cells (all pre-2026-10-23 sparse KL numbers were void because of it). v2 refs use 1 GiB blocks + JDE prefill.
- **Native EXL3 load verification** (`JI_EXL3_CHECK=<oracle.gguf>`): loads a checkpoint directory (`.../config.json +
  model.safetensors.index.json`) and the oracle GGUF in parallel, compares every tensor (bytes and metadata keys),
  exits with count of matches and any differences to stderr. Sage-A1.5: 1475 tensors identical except ssm_a (2 ulp).
- **Decode speed**: `~/jt/mtpsrv.sh` (production config; `NMAX=0` no MTP) + `~/jt/spectest.py` (3 prompts x
  1024 tokens); depth: `scripts/depth_kv.sh N` (restores saved KV from `/mnt/ssd/kvcache`). **Prefill**:
  `scripts/depth_bench.py`, `~/jt/dpb.py`.
- **Profiling**: `JI_PROFILE=N` (per-token phases, expert-cache hits/misses per step size), `JI_PROFILE_GPU=1`
  (per-kernel GPU time; disables replayed command buffers), `JI_TP_PROF=N`, `JI_TP_GPROF=1` (TP helper),
  `JI_PF_GPROF` / `JI_PF_TL` (prefill).
- **VRAM over-commit check**: `/sys/class/drm/card{0,1}/device/mem_info_{vram,gtt}_used`. Over-commit does not
  fail; the driver silently evicts buffers to GTT and unrelated kernels run 2x slower (see KVarN below).
- Knobs: `engineering-SIREN/current/code-map.md` lists every `JI_*` environment variable.

## 8. Feature status

| feature | status | doc |
|---|---|---|
| JDE decode engine (one-shot replayed tokens, fused kernels, EXL3 mat-vecs) | default on | features/jde-decode-engine.md |
| Expert memory hierarchy (pinned host, VRAM cache, elastic slots in KV tail + compute buffer) | default on | features/expert-cache.md |
| Tensor-parallel decode + prefill on two GPUs (TP2) | production | features/tensor-parallel-two-gpus.md |
| MTP speculative decoding (draft head on the helper GPU) | production (serve_prod / boot) | features/mtp-speculative-decoding.md |
| Sparse attention, decode + prefill | default on | features/sparse-attention.md |
| KVarN-6 KV cache (6-bit in place, `JI_KVARN=1`) | **opt-in, not production-ready** | features/kvarn6-kv-cache.md |
| EXL3 kernels | default on | exl3-support.md |

## 9. Open work, in priority order

0. **TOP PRIORITY (user, 2026-10-26): native EXL3 — load exllamav3 checkpoints without conversion, keep their MTP
   layer and vision tower, make the current `mul1` codebook fast.** Full step-by-step plan, pitfalls, checks and
   remaining headroom: [`engineering-SIREN/current/native-exl3-plan.md`](engineering-SIREN/current/native-exl3-plan.md).
   Plan order (user-decided, 2026-10-30): 1. **Post-prefill decode slowdown [DONE 2026-10-31, engine 104fa96]**: huge-page host imports; decode now within 1% of fresh after a 21.7k prefill. 2. **Fast mul1 decode [DONE 2026-10-31, engine d44ea8d]**: Sage 46.4 / 55.1 / 49.7 t/s (was 39.6 / 46.0 / 41.9), prefill 557 t/s (was 541). 3. **Native EXL3 loader [DONE 2026-10-31, engine 479f8ab]**: checkpoint directory loads directly, no GGUF conversion, 1475 tensors byte-identical vs oracle (ssm_a 2 ulp). 4. **Sage's own MTP head [DONE 2026-10-31, engine f89ed0e]**: IQ4_NL re-quant at load, p-min 0.8 lossless, Sage +22/+41/+36% (47.4/62.8/55.3 t/s), acceptance 87/95/91%. 5. **MTP verify-step anomaly [CONCLUDED 2026-10-31]**: no bug; per-step decode cost 19.7 / 35.8 / 45.7 ms for 1 / 2 / 3 tokens is MoE-bound (each token routes to mostly different experts). 6. **Prefill two-card breakdown and balance [CONCLUDED 2026-10-31]**: no cheap win; helper GPU busy 99.7 %, primary waits ~0.9 s per 4k-token batch; mixer all-reduce over the helper's PCIe 2.0 x4 is the limiter (6.4 s vs 5.0 s on the primary). Cards equal in ALU (292 vs 293 Gdecode/s). 7. **EXL3 GEMM energy rewrite [PARTLY DONE 2026-10-31, engine f0dcb49]**: narrow expert tiles (pf_mv_id.comp) for remainders <= 8 rows. Correctness (L_code 32k TP2, tail via JDE prefill 256 tokens, engine f0dcb49): `JI_PF_NARROW=0` KL 0.00015 / top1 0.9990, `JI_PF_NARROW=1` KL 0.00012 / top1 1.0000. Prefill speedup: 500 tokens -5 to -7 %, >=1000 tokens ~1-3 %. Remaining: large-tile overhead (~28 % of MoE GEMM time, ~5-7 % of prefill) needs VALU work per k-step cut; estimated 10-15 % win. 8. **Short-prompt prefill (no re-stream) [DONE 2026-10-31, engine 14eb6fe, f519a49]**: llama-server was splitting chat prompts at the last user message start; fixed with new `JI_CKPT_SPLIT_MIN` (default 2048). Prefill t/s: 1159 tokens 208 -> 420, first chat turn 5564 -> 4898 ms. Decode unchanged. 9. KVarN-6 fixes. 10. Sparse-attention track. 11. Vision.
   It supersedes prior items below. Current items continue (or in parallel where independent).

1. **KVarN-6: prefill from scratch beyond ~64k is wrong - CONFIRMED REAL (2026-10-31, job88)**: through the
   production JDE prefill (TP2, default sparse, harness now with `JI_JPF=1`, 32 JDE prefill batches, 0 graph
   fallbacks) 130k from scratch: q8_0 KL 0.00123 / top1 0.9883 (= the documented end-to-end 0.00125), KVarN
   **0.118** / top1 0.906 (the old recorded TP2 value). 32k from scratch is fine (single card, sparse off: q8 0.00071,
   KVarN 0.00092). Restored states decode fine at every length (32k 0.00058, 64k 0.00076, 130k 0.00074-0.00121).
   So the defect is in the JDE prefill's KVarN path (materialise / copy-out / seal) somewhere between 32k and 130k.
   Next: bisect by depth (64k / 98k rows from scratch with `JI_JPF=1`), then `kvrblk.py` on a saved from-scratch
   state vs the reference (which layers / cells go wrong first).
   Detour of 2026-10-31 (do not repeat): the KL harness ran the graph-path prefill (no `JI_JPF`), which never builds
   KVarN records -> KL ~2 and empty saved states; all knobs tried on that (`JI_PF_CBALL`, `JI_ELASTIC=0`,
   `JI_STATIC=0`, `JI_KVR_PFT=1`, in-process reload) were meaningless.
2. **KVarN-6 decode at 200k is slower**: 25.1 / 27.4 t/s vs q8 32.4 (VRAM spill ruled out after the ring fix).
   Next: `JI_PROFILE_GPU=1` at 200k, compare `attn_kvr6q` + `attn_kvr6` vs `attn_hm`.
3. Then KVarN K6V5 / K6V4 asymmetric variants (user-approved), end-to-end gate, make it default if it wins.
4. **Stability**: S10 TP2 prefill soak (`JI_PF_XQ=0` to separate a sync race from hardware); after the PSU swap:
   145 W re-verify, the power-off question, SSD adapter.
5. **MVP hardening**: full API suite (needs pip permission), single-card release config, graph-path KV-loss root
   cause (or block that path), docs for server / prompt reuse / system tuning.
6. **Throughput toward targets**: decode — expert misses stream over PCIe (hit rate ~84 %; MoE mat-vecs are ~50 %
   of a 2-token step's GPU time); prefill — power-bound at 100 W, EXL3 GEMM efficiency ~45 % of peak.
   `open-paths.md` has the ordered list.
7. **Sage-A1.5 follow-ups:** (a) its own MTP head — dequantise the EXL3 `mtp.*` tensors (numpy mul1 decoder in
   `tools/exl3/exl3_ref.py`) and build the IQ4_NL sidecar like `mtp-IQ4_NL.v2.gguf` (`tools/mtp_extract.py`), or
   fetch only the BF16 `mtp.*` shard of InternScience/Sage-A1.5 (download needs permission); expected +14-23 %
   decode as on Qwen3.6. (b) a faster `mul1` mat-vec (open-paths #7) recovers the ~20 % gap. (c) the KL references
   are Qwen3.6's: Sage has no quality gate yet.
8. **Post-MVP sparse-attention track** (user-approved order): micro-block / sub-page centroids (M1-M2) before any
   learned estimator; learned corrector with warm start, <= 24 h of GPU time total. `open-paths.md` L1-L10, M1-M5, T0.

## 10. Lessons that cost time (don't repeat)

- Wrong references poison everything: check reference states for zero rows; keep refs on a known-good path.
- VRAM over-commit is silent (GTT eviction). 130 MiB of extra buffers halved MoE speed under MTP (2026-10-24/25).
- A "perfectly invariant" knob is usually disconnected: smoke-test new flags first.
- The model's global attention is broad: sparse quality depends on the fraction read exactly, not on ranking
  cleverness (oracle ranking is at most ~1.7x better).
- Single-card runs on a two-card box: the remaining card's Vulkan index changes when the other is missing.
- Big page cache after downloads costs ~10 % decode at depth; CPU governor schedutil costs more.
- In the agent's shell tool, chained `sleep` waits get blocked: use background commands with an until-loop.
- On amdgpu, every host buffer imported into a GPU (userptr) costs page-walk time on every submit; use huge pages
  for imports and keep them few (2026-10-31: prefill exchange buffer 134 MiB = 33k 4 KB pages = 23 ms of kernel
  overhead per submit; 2 MiB alignment + MADV_HUGEPAGE + pre-fault cuts that to 0.1 ms).
- A per-batch fixed cost is not necessarily transfer: measure the kernel (JI_PF_VRAMX removed the PCIe upload for 15 % of
  experts and changed nothing).
- **A KL check is only as good as the path it runs:** confirm the JDE prefill actually ran (count "jde prefill TL rank 0" lines with `JI_PF_TL=1 KLX="-lv 4"`). The harness ran the graph-path prefill for weeks.
- **Stop the boot service with `~/jt/stopsrv.sh`**, never `stop + sleep`. The service may still be starting; stopsrv waits until all daemons exit.

## 11. Document map

- `README.md` — user-facing: what it is, how to run, clients, numbers.
- `engineering-SIREN/index.md` -> `current/STATE.md` (live state), `open-paths.md` (every open lever, ordered),
  `features/` (one doc per feature, stable), `code-map.md` (engine source map + all knobs), `engineering-log.md`
  (chronological, 1700+ lines; newest at the end), `candidate-ledger.md`, `physics.md`, `quality-gates.md`,
  `decode-engine.md`, `dual-card-design.md`, `tp-prefill-design.md`, `exl3-support.md`.
- `tools/aux-jt/README.md` — the AUX-1 test harness.

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

## 2. Current state (2026-10-26)

**Production = boot service on AUX-1** (`junkinfer.service` -> `scripts/serve_boot.sh` -> `scripts/serve.sh`):

- `http://192.168.1.4:8080/v1` (OpenAI: chat / completions / responses; Anthropic: `/v1/messages`), web UI at `/`.
  No API key (set `JI_API_KEY` in `/etc/junkinfer.env`). Model alias `qwen3.6-35b-a3b`.
- Two cards enumerated: TP2 tensor-parallel decode + two-card JDE prefill + MTP speculative decoding
  (= `scripts/serve_prod.sh`). One card: single-card JDE. q8_0 KV, 262,144 context, sparse attention on,
  thinking on, **Qwen-Sharp chat template** (`ref/templates/qwen_sharp.jinja` = upstream
  `peculiar-ragdoll/Qwen-Sharp-Chat-Templates` `chat_template.jinja`, identical as of 2026-10-26).
- Restarts on exit. Verified across two reboots on 2026-10-26: server, thermal guard, fan curve, governor all up.
  `scripts/smoke_api.py`: ALL PASS (chat, tool-call parsing, prefix cache 6839 -> 23 tokens, prefill 536 t/s).

**Sage-A1.5 (boot default, 2026-10-26):** TP2, **no MTP**, Qwen-Sharp template, alias `sage-a1.5`. smoke_api ALL
PASS (prefill 541 t/s on 6.8k). Decode rewrite / story / explain 39.6 / 45.0 / 41.7 t/s; Qwen3.6 same config and day:
43.2 / 51.2 / 48.0. Why slower: Sage's EXL3 uses the `mul1` codebook (Qwen3.6: `mcg`); same bit widths, same expert
hit rate (93.6 vs 93.9 %), but GPU time 21.8 vs 18.0 ms/token — our `mul1` mat-vec costs ~20 % more per weight on
gfx803 (open-paths #7). MTP: the converter drops Sage's own (EXL3) MTP layer; the Qwen3.6 draft head gains nothing
on Sage (34.6 / 45.6 / 42.6 with vs 38.6 / 45.1 / 41.7 without; rewrite acceptance 64 % vs 94 % on Qwen), so
serve_boot.sh turns MTP off for Sage. TP2 hot-expert set regenerated for Sage (`ref/tp/tp_hot_sage15_exl3.bin`;
top-80 overlap with Qwen's only 36 %; no measurable speed change). Sage quality (KL vs its BF16) is not measured.

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

AUX-1 paths: models `/mnt/ssd/models/` (run models only from the SSD), saved KV / states `/mnt/ssd/kvcache/`
(KL prefix states `jref_v2/`), harness `~/jt/` (snapshot in `tools/aux-jt/`), engine `~/junkinfer/engine`
(rsynced, **not a git repo**), project `~/junkinfer` (scripts, ref, logs).

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
   Plan order (user-decided, 2026-10-30): 1. **Post-prefill decode slowdown [DONE 2026-10-31, engine 104fa96]**: huge-page host imports; decode now within 1% of fresh after a 21.7k prefill. Sage 39.6 / 46.0 / 41.9 t/s. 2. Fast mul1 decode (open-paths #7). 3. Native EXL3 loader (plan steps 1-3). 4. Sage's own MTP head. 5. MTP verify-step anomaly. 6. Prefill: two-card time breakdown + balance. 7. EXL3 GEMM energy rewrite. 8. Short-prompt prefill (no re-stream). 9. KVarN-6 fixes. 10. Sparse-attention track. 11. Vision.
   It supersedes prior items below. Current items continue (or in parallel where independent).

1. **KVarN-6: prefill from scratch beyond ~64k is wrong** (130k: KL 0.27 single card, 0.118 TP2; 32k 0.00085 and
   64k 0.00115 are fine; decode from restored states is fine everywhere: 0.0005-0.0011).
   Found so far (2026-10-25): not the separate prefill copy buffer (`JI_KVR_PFT=1` at 32k = same KL). A from-scratch
   130k state vs the reference (`kvrblk.py`): layer 3 (first attention layer) exact everywhere (1.8 % = KVarN
   noise); later layers deviate from cell **12288** on (the first ubatch that attends over materialised sealed
   records), moderately (rel. error 0.02-0.19), and the recurrent state drifts 0.3-7.7 %. That looks like
   compounding quantisation noise, not a broken region — yet the final KL is 200x the 64k value.
   **Next step: `~/jt/job40.sh`** (written, not run; state kept in `/mnt/ssd/kvcache/tmp_j36`): restore that
   from-scratch state and run the tail with `JI_KVARN=1` and `=0`. Bad KL with both -> the state is bad (then bisect
   by depth: 64k / 98k rows from scratch; check `jde_kvr_pf_materialize` / `kvr_mat` / the seal at each ubatch
   end in `jde_prefill.inc`); good KL -> the bug is in the in-process tail evaluation after a long prefill.
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

## 11. Document map

- `README.md` — user-facing: what it is, how to run, clients, numbers.
- `engineering-SIREN/index.md` -> `current/STATE.md` (live state), `open-paths.md` (every open lever, ordered),
  `features/` (one doc per feature, stable), `code-map.md` (engine source map + all knobs), `engineering-log.md`
  (chronological, 1700+ lines; newest at the end), `candidate-ledger.md`, `physics.md`, `quality-gates.md`,
  `decode-engine.md`, `dual-card-design.md`, `tp-prefill-design.md`, `exl3-support.md`.
- `tools/aux-jt/README.md` — the AUX-1 test harness.

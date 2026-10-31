# Native EXL3: load checkpoints directly, keep MTP and vision, fast mul1 — implementation plan

Status: **ACTIVE — TOP PRIORITY (user, 2026-10-26).** Nothing below is implemented yet. Written for the next engineer
or LLM agent; read `../../HANDOFF.md` first (access, rules, workflow, measurement).

## 0. Goal and acceptance criteria

The user wants three things, in this order of importance:

1. **No conversion step.** `serve.sh` points at an exllamav3 directory (e.g. `~/models/Sage-A1.5-exl3-4.0bpw/`, the
   unmodified Hugging Face download) and the server runs it. No GGUF copy of the weights on disk.
2. **Nothing dropped.** The checkpoint's MTP layer drives speculative decoding (no borrowed Qwen3.6 sidecar), and its
   vision tower works for image input.
3. **The newest EXL3 format runs as fast as possible.** Current exllamav3 quants (1.5.0, codebook `mul1`) must be at
   least as fast as the older `mcg` quants are today.

Done means, measured on AUX-1 (rules: `HANDOFF.md` §4, §7):

| # | check | pass |
|---|---|---|
| A1 | Tensors loaded natively vs the converter's GGUF (Qwen3.6 and Sage) | byte-identical, every tensor (§3 step 1) |
| A2 | Greedy outputs, native vs GGUF | identical md5 on `spectest.py` prompts; KL rows (`kl_v2.sh`) unchanged |
| A3 | Load time | <= today's GGUF load + 60 s (page cache dropped) |
| A4 | MTP from the checkpoint | Sage acceptance >= ~85 % on story / explain, decode faster than without MTP |
| A5 | Vision | an image question answered correctly through the API (Sage and Qwen3.6), text-only speed unchanged |
| A6 | mul1 speed | Sage decode GPU ms/token <= Qwen3.6 mcg's (today 21.8 vs 18.0 ms) |
| A7 | No regression | `smoke_api.py` ALL PASS; 200k decode and 151k prefill within 2 % of today |

## 1. Background you need

**What "EXL3 natively" already means.** The engine *executes* EXL3 natively: ggml has `GGML_TYPE_EXL3_2..8`, JDE's
`x3_mv.comp` (decode mat-vec), the prefill GEMM (`mul_mm_gcn_exl3` in ggml-vulkan, JDE `pf_mm`) and the CPU scalar
path decode the trellis at run time. Bits are never re-quantised. What is not native is the **container**: the
weights go through `tools/exl3/exl3_to_gguf.py` (safetensors -> GGUF) because llama.cpp's model loader only reads
GGUF. That converter also **drops the MTP layer and the vision tower** — the real defect.

**What the converter does** (you must reproduce each of these, or make the consumer not need it):

| # | transform | why |
|---|---|---|
| T1 | trellis tiles `[K/16][N/16][16*bits/16 int16]` -> `[N/16][K/16][...]` (transpose of the tile grid) | a group of 16 outputs contiguous: coalesced loads in `x3_mv` / GEMM |
| T2 | GDN value heads grouped -> tiled order (`block_perm`): whole 128-wide Hadamard blocks of `in_proj_qkv` (N side, V part), `in_proj_z` (N), `out_proj` (K) move, with their `suh`/`svh`; `in_proj_a/b`, `A_log`, `dt_bias`, `conv1d` V rows permuted the same way | llama.cpp's qwen35 graph and JDE's `gdn_*` kernels expect the tiled order (as `convert_hf_to_gguf.py` produces) |
| T3 | RMSNorm weights +1 (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, final `norm`); NOT `linear_attn.norm` | Qwen3.5 stores norm weights as offsets |
| T4 | `A_log` -> `-exp(A_log)`; `conv1d` squeezed to `[channels, 4]` | llama.cpp's ssm conventions |
| T5 | routed-expert `suh`/`svh` kept fp16 and stacked `[E, K]` / `[E, N]`; experts stacked `[E, N, K*bits/8]` | one tensor per projection per layer (`ffn_*_exps`) |
| T6 | head: rows past the vocabulary dropped in whole 16-row tiles; embeddings bf16, zero rows up to n_vocab | n_vocab = head rows (multiple of 128 for the output Hadamard) |
| T7 | tokenizer / hparams / chat template: copied from an existing qwen35moe GGUF, tokens re-derived from the checkpoint's `tokenizer.json` when they differ | llama.cpp needs `tokenizer.ggml.*` and `qwen35moe.*` keys |
| T8 | codebook detected per tensor (`.mcg` / `.mul1` marker tensors), must be uniform -> `junkinfer.exl3.codebook` (0 mcg, 1 mul1); MCG multiplier asserted `0xCBAC1FED` | kernels take the codebook as a specialisation constant |

Read `tools/exl3/exl3_to_gguf.py` completely before starting (it is ~280 lines); `tools/exl3/exl3_ref.py` is a
numpy reference decoder (mcg + mul1, any width) — use it for every correctness check below.

**Checkpoint layout (exllamav3 1.5.0, Sage-A1.5 as the example):** 5 safetensors shards + `model.safetensors.index.json`;
`config.json` (`Qwen3_5MoeForConditionalGeneration`, `text_config`, `vision_config`); `quantization_config.json`
(40 MB: `bits`, `head_bits`, `mtp_bits`, `vision_bits`, `codebook`, `tensor_storage` per module). A quantised linear
`X` is four tensors: `X.trellis` (int16), `X.suh` (fp16 [K]), `X.svh` (fp16 [N]), and a marker `X.mul1` or `X.mcg`.
Unquantised tensors stay bf16/f32 (`X.weight`): per text layer `mlp.gate`, `mlp.shared_expert_gate`,
`linear_attn.in_proj_a/b`, `A_log`, `conv1d`, `dt_bias`, `linear_attn.norm`, the layernorms. Prefixes:
`model.language_model.*`, `lm_head.*`, `mtp.*` (one full block: `fc`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`,
`layers.0.{self_attn.{q,k,v,o}_proj, q_norm, k_norm, mlp.{experts.N, shared_expert, gate, shared_expert_gate},
input_layernorm, post_attention_layernorm}`, `norm`), `model.visual.*` (27 blocks, hidden 1152, MLP 4304, merger).

**Oracles you already have** (do not delete them; they are the test fixtures):
- `/mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf` (mcg) and `/mnt/ssd/models/Sage-A1.5-exl3-4.00bpw.gguf` (mul1):
  converter output = what the native loader must reproduce byte for byte.
- `~/models/Sage-A1.5-exl3-4.0bpw/` (HDD): the Sage checkpoint. The Qwen3.6 EXL3 checkpoint the community GGUF came
  from is not on AUX-1 any more — check `~/models` / VENE; re-download needs the user's permission.
- `/mnt/ssd/models/mtp-BF16.gguf`: Qwen3.6's MTP block in GGUF naming (`blk.40.nextn.eh_proj`, `nextn.enorm`,
  `nextn.hnorm`, `nextn.shared_head_norm`, `attn_q/k/v/output`, `ffn_*_exps`, ...): the name / shape mapping oracle.
- **Not an oracle:** `/mnt/ssd/models/mmproj-Q8_0.gguf` is MiniCPM-V 4.7's projector (`clip.projector_type
  minicpmv4_7`), not Qwen3.6's. There is no Qwen3.6 / Sage vision GGUF on the box.

## 2. Architecture decision

Two ways to "load natively":

- **(a) Load-time transform (recommended).** A safetensors reader inside llama.cpp builds the same `gguf_context`
  (metadata + tensor infos) the converter would have written, and fills tensor data by reading the shards and applying
  T1-T6 on the fly while uploading. Every consumer (llama.cpp graph, JDE, ggml-vulkan) is unchanged. Load cost: the
  tile transpose and permutations are cheap memory moves; I/O dominates (19.4 GB). The experts are copied into pinned
  host memory anyway (`--load-mode none`), so mmap zero-copy is not lost.
- **(b) Consumers read exllamav3's layout directly** (no T1/T2: kernels index `[K/16][N/16]`, GDN kernels take grouped
  order). Saves the load-time transform but touches every EXL3 kernel and the GDN kernels, and llama.cpp's graph
  path. Not worth it: do (a). Revisit (b) only if A3 fails.

Do it in C++ in the engine fork (no Python at run time). Where:
- `src/llama-model-loader.cpp`: the constructor calls `gguf_init_from_file(fname, ...)` (line ~571) and has a
  `gguf_init_from_file_ptr` variant (~683). Add: if `fname` is a directory containing `config.json` +
  `quantization_config.json` (or a `.safetensors` index), call a new `exl3_open(dir)` that returns a synthetic
  `gguf_context` plus a per-tensor *producer* (callback that writes the tensor's bytes into a destination buffer).
- The loader's data path (`load_all_data`, the mmap / file-read branches) must call the producer instead of
  reading at a file offset. Find every place that assumes `weights_map` entries point into a GGUF file (offset +
  `llama_file`); give EXL3 entries a distinct kind.
- New files: `src/llama-exl3-loader.{h,cpp}` (keep it out of `src/junkinfer` — it is a loader, not JDE).

## 3. Step-by-step

Each step ends with a check; do not start the next before it passes. Commit after each step (engine fork, branch
`junkinfer`), rsync, build on AUX-1, stop `junkinfer.service` before any run.

### Step 1 — safetensors reader + tensor plan, verified offline (1-2 days)

1. Port the converter's *plan* to C++: for each GGUF tensor name, shape and ggml type, the list of source tensors and
   the transform (T1-T6). Parse `config.json` and `model.safetensors.index.json` with the JSON library llama.cpp
   already vendors (`vendor/nlohmann/json.hpp` / common). Read shard headers (8-byte little-endian length + JSON);
   mmap shards read-only.
2. Implement the producers: tile transpose (T1), Hadamard-block permutations (T2), +1 norms (T3), `-exp` / squeeze
   (T4), expert stacking with fp16 scales (T5), head trim + embedding pad (T6). Bits per weight = `trellis.shape[-1]
   / 16` (a 16x16 tile = 256 weights = 16 x bits int16 values); derive it as the converter does (`exl3_parts`) and
   assert it matches `quantization_config.json`.
3. **Test harness first:** a small tool `llama-exl3-dump DIR NAME` that runs the producer for one tensor and writes
   its bytes. Compare against the converter GGUF for **every** tensor of both models (sha256 per tensor; a Python
   script reading the GGUF with `gguf-py`). Must be byte-identical (A1). This catches 90 % of the bugs before any GPU
   run.

Pitfalls:
- bf16 in safetensors vs F32 in the GGUF: the converter writes norms / small tensors as F32 and embeddings as BF16;
  match the converter, not your intuition.
- Shard boundaries: one module's four tensors can sit in different shards.
- Endianness / alignment: safetensors data starts right after the header, unaligned; copy, don't type-pun.
- Memory: the box has 32 GB with ~17.5 GB pinned for experts while serving. Produce tensor by tensor (streaming); never
  hold a full shard decoded in RAM. Only one model process at a time (`HANDOFF.md` §4).

### Step 2 — metadata without a template GGUF (1 day)

The converter copies hparams and the tokenizer from an existing GGUF (T7). Native loading must derive them:
1. hparams from `config.json` `text_config` -> the `qwen35moe.*` keys. Write the mapping table by reading
   `convert_hf_to_gguf.py` (class for `Qwen3_5MoeForConditionalGeneration` / qwen35moe) and diffing against the keys in
   the existing GGUFs (`gguf-py` dump of `Qwen3.6-35B-A3B-exl3-4.00bpw.gguf`). Include rope (partial rotary 0.25,
   theta 1e7, M-RoPE sections `[11, 11, 10]`, interleaved), GDN dims, expert counts, `nextn_predict_layers`.
2. Tokenizer from `tokenizer.json`: BPE vocab (`model.vocab`), merges (`model.merges`), added tokens with their
   special flags, token types (as the converter's `over` block does), pre-tokenizer = the qwen35 one, BOS/EOS/PAD from
   `tokenizer_config.json` + `config.json`. Vocabulary padded to the head's rows ([PAD] tokens).
3. Chat template from `chat_template.jinja` (serve.sh overrides it with Qwen-Sharp anyway).
4. Check: dump all KV pairs of the synthetic context and diff against the converter GGUF's KV (ignore
   `junkinfer.exl3.source` and `general.*` names). Tokenise a few thousand lines of `docs_all.txt` both ways:
   identical ids.

Pitfall: llama.cpp computes the pre-tokenizer from a hash in the Python converter; in C++ just set the known qwen35
pre-tokenizer string the existing GGUF carries.

### Step 3 — load and run the text model natively (1 day)

1. Wire `exl3_open` into the loader (§2). `serve.sh`: accept a directory as the model; `JI_MODEL` default becomes the
   checkpoint directory on the SSD (see disk note below).
2. Checks (A2, A3, A7): greedy `spectest.py` md5s equal to the GGUF run; `kl_v2.sh L_code 0` and `1` equal (Qwen3.6
   only if its checkpoint is available; else Sage native vs Sage GGUF via `llama-junk-ref` compare against a ref made
   from the GGUF); load time; smoke_api.

Disk: `/mnt/ssd` has ~25 GB free. The Sage checkpoint (19.4 GB) must move to the SSD (run models only from the SSD:
HDD loads are minutes). Make room by **moving** (never deleting without the user's approval) old GGUFs / tmp files to
`~/models/ssd-overflow/` — candidates: `MiniCPM-V-4.7-*.gguf` (17 GB), `head-*.tmp.gguf`, `qwen25-1b`, `qwen3-06b`,
`/mnt/ssd/kvcache/tmp_j36` (KVarN diagnostic state, 1.5 GB). Keep the converter GGUFs until A1/A2 pass.

### Step 4 — MTP from the checkpoint (2-3 days)

Today: `-md /mnt/ssd/models/mtp-IQ4_NL.v2.gguf --spec-type draft-mtp` loads a separate sidecar GGUF; JDE runs the
draft head on the helper GPU (`jde_mtp_*` in `jde.cpp` ~2514-2941; head re-quantised to the 40,960-token draft
vocabulary, `JI_MTP_HEAD_TYPE=iq4_nl`, `JI_MTP_VOCAB_FILE`).

1. Map `mtp.*` to the nextn block `blk.<n_layer>.*` (= `blk.40`), names and shapes from `mtp-BF16.gguf`:
   `mtp.fc` -> `nextn.eh_proj`; `pre_fc_norm_embedding` -> `nextn.enorm`; `pre_fc_norm_hidden` -> `nextn.hnorm`;
   `mtp.norm` -> `nextn.shared_head_norm`; `layers.0.self_attn.{q,k,v,o}_proj` -> `attn_q/k/v/output` (q includes
   the output gate: Qwen3.5 `attn_output_gate`); `q_norm/k_norm` -> `attn_q_norm/attn_k_norm`;
   `layers.0.mlp.*` -> `ffn_*_exps`, `ffn_*_shexp`, `ffn_gate_inp`, `ffn_gate_inp_shexp`;
   `input_layernorm/post_attention_layernorm` -> `attn_norm/post_attention_norm`. Verify every name/shape against the
   oracle; apply T3 (+1) to the RMSNorms exactly as the converter does for the main layers (check the BF16 oracle:
   are its nextn norms already +1?).
2. Serve the MTP block from the same model (no `-md` sidecar): llama.cpp's `load_mtp` (common.cpp ~1731) loads the
   nextn block when `--spec-type draft-mtp` is given; check how the sidecar path and the in-model path differ in
   `common/speculative.*` and pick the in-model path.
3. JDE: `jde_mtp_upload` uploads the block's tensors as they are; EXL3 types must go through the `x3_mv` path like the
   main layers (check the MTP decode uses `e->mv` with type dispatch — it was written for IQ4_NL/BF16 tensors).
4. **The head trap:** the draft head is a *row subset* of the LM head (40,960 rows). An EXL3 head cannot be
   row-sliced: its output Hadamard (`svh`, 128-row blocks) mixes rows. Do as today: dequantise the 6-bit EXL3 head on
   the CPU once at load (exl3_ref-equivalent in C++, or ggml's EXL3 `to_float`), take the draft rows, quantise to
   IQ4_NL (`JI_MTP_HEAD_TYPE`). Cache the result (a small file next to the checkpoint is fine; it is not a weights
   conversion).
5. The draft vocabulary file (`ref/mtp/hyperqwen_draft_vocab_40960.json`) is Qwen-tokenizer ids: valid for Sage
   (identical tokenizer). For other models, check the tokenizer first.
6. Checks (A4): acceptance and t/s with `spectest.py` (3 x 1024 tokens); greedy md5 identical with MTP on / off
   (speculation is lossless; a difference = bug). Expected from Qwen3.6: +14-23 % decode.

### Step 5 — vision (3-5 days)

llama.cpp runs vision through libmtmd (`tools/mtmd`, `clip.cpp`) from a separate `--mmproj` GGUF. Qwen3.6 / Sage
vision: Qwen3.5-VL style tower (27 blocks, hidden 1152, MLP 4304, patch 16, spatial merge 2, no deepstack), closest
llama.cpp projector `PROJECTOR_TYPE_QWEN3VL` (`clip.cpp` ~1000, ~1740, ~4195).

1. Get a known-good reference first: ggml-org's Qwen3.6 mmproj (`ggml-org/Qwen3.6-35B-A3B-GGUF`, Q8_0 or F16 — the user
   approved this download on 2026-10-20; confirm again) and check image answers through the server with the Qwen3.6
   text model. This proves the clip graph and the server path before EXL3 enters.
2. Native: give `clip.cpp` the same synthetic-loader treatment (its loader also reads GGUF: `clip_model_loader`),
   mapping `model.visual.*` to the mmproj tensor names of the reference file.
3. The tower is EXL3 6 bpw. Options: (i) dequantise to F16 at load (~0.4-0.5 B params: ~0.9 GB F16 — too much VRAM on
   the primary; put the tower on the helper GPU or CPU) or (ii) keep EXL3 and run clip's mat-muls through ggml's EXL3
   mul_mat. Start with (i) on the CPU / helper for correctness, then measure.
4. **Dimension trap:** the vision MLP width 4304 is not a multiple of 128 (EXL3's Hadamard block). Check the stored
   `suh`/`svh` lengths and trellis shape of `linear_fc1/fc2`: exllamav3 must pad; the producer must reproduce
   exactly what the exllamav3 forward does (crop after the output Hadamard). Also: blocks carry both `attn.qkv.weight`
   and `attn.{q,k,v}_proj.*` — find out which one exllamav3 uses (likely the split, quantised ones) before mapping.
5. **M-RoPE trap:** image tokens get 3-D positions (t, h, w; sections 11/11/10, interleaved). JDE's `attn_qk` /
   `pf_qk` implement 1-D RoPE; for pure text the three components are equal, so text is fine. A ubatch containing
   image embeddings must not go through JDE as is: either implement M-RoPE in `pf_qk` (positions per section from the
   ubatch's `pos` array, which llama.cpp fills with 4 * n_tokens entries for M-RoPE) or route such ubatches to the ggml
   graph path. **But the graph path loses layer-3 KV writes with 3.75 GiB Vulkan blocks** (engineering log
   2026-10-23): if you route there, run with 1 GiB blocks or fix that bug first. Prefer implementing M-RoPE in JDE
   prefill (decode stays 1-D: generated tokens are text).
6. Checks (A5): image Q&A correct on a few test images (vs the reference mmproj on Qwen3.6); text-only decode and
   prefill speed unchanged; VRAM over-commit check (`mem_info_vram_used` / `gtt_used`, see §5).

### Step 6 — fast mul1 (1-2 days; can run in parallel with steps 1-5)

**DONE 2026-10-31 (engine d44ea8d):** exact mul1 byte sum (dropping f16 round trip, using `dotPacked4x8EXT`) + affine fold reach mcg's 293 Gdecode/s. Sage decode 46.4 / 55.1 / 49.7 t/s (was 39.6 / 46.0 / 41.9). v_sad_u8 is not emitted by ACO from GLSL on gfx8; the gain came from ALU path optimization, not a single instruction. See engineering-log 2026-10-31.

Why it matters: current exllamav3 quants (MiniCPM-V 4.7, Sage-A1.5: 1.5.0) use the `mul1` codebook; the community
Qwen3.6 quant uses `mcg`. Confirm in the exllamav3 source which codebook is the current default before claiming it
for all new quants.

The per-weight decode (`shaders/x3_mv.comp`, `mcg()` / `mul1()` ~lines 131-145; the same functions in the GEMM and
dequant shaders in ggml-vulkan):
- mcg: `x = v * 0xCBAC1FED` (32-bit mul: quarter rate on GCN), and, xor, two f16->f32, one add.
- mul1: `x = v * 0x83DCD12D`, byte sum of x by masks / shifts / adds, +1024, int->float, fma, f16 round trip.

Plan:
1. **Look at the ISA first** (hours). `RADV_DEBUG=shaders` (or `ACO_DEBUG`) dumps the compiled gfx803 code; build a
   tiny test shader in `lab/matvec/` with each candidate and count VALU instructions per weight. Candidates:
   - `v_sad_u8(x, 0, 0)` = byte sum in one full-rate instruction on GCN3. GLSL has no intrinsic. Try
     `GL_EXT_integer_dot_product`'s `dotPacked4x8EXT(x, 0x01010101u)` and check whether ACO lowers it to `v_sad_u8`
     or to shifts. Also try spelling the sum as `abs`-difference patterns NIR might recognise (`nir_op_sad_u8x4`).
   - `v_cvt_f32_ubyte0..3`: four full-rate byte->float conversions + 3 adds (float domain; the +1024 and scale fold
     into constants): `unpackUnorm4x8(x)` * 255 may lower to exactly these.
   - fold `+1024` into the fma constants (`fma(s, a, b + 1024 a)`), keep exactness in f32.
   - the f16 round trip (`float(float16_t(...))`) exists to match exllamav3 bit-exactly. Dropping it saves 2 ops but
     changes numerics: KL gate required (expect ~0). Keep it if the gate is not clean.
   - a 1021-entry LUT in LDS indexed by the byte sum replaces int->float + fma + round trip (one LDS read; LDS is
     already busy in `x3_mv` — measure).
2. Implement the best variant for `x3_mv` (decode) first; per-kernel time with `JI_PROFILE_GPU=1`
   (`x3_mv_moe exl3_4 512x2048` etc.), then end to end (`spectest.py`), then KL.
3. If ACO never emits `v_sad_u8` and the remaining gap is large: Mesa is the system driver (RADV); building and using
   our own Mesa (or patching an ACO pattern) is possible but a bigger commitment (deployment on the box, upgrades).
   Ask the user before going there.

Correction to an earlier claim: **prefill gains +2.6 %** (543 -> 557 t/s at 6.8k). The prefill GEMM decodes each weight once per tile and
reuses it across the ubatch's 4096 tokens, so the codebook ALU cost is amortised but still present; the fold applies
to every tile. MTP verify steps (2 tokens) amortise decode's per-token decoding; the fold helps there too.

## 4. Things to watch for (all steps)

- **Quality gate before speed claims** (`SIREN.md`). Every change to kernels or numerics: KL on the v2 rows. Native
  loading must be bit-identical (A1/A2), not "close".
- **VRAM over-commit is silent**: the driver evicts buffers to GTT and unrelated kernels run 2x slower (KVarN rings,
  2026-10-25). After adding MTP weights, vision weights or a head cache, check `/sys/class/drm/card{0,1}/device/
  mem_info_{vram,gtt}_used` before and after a run; `JI_TP_MARGIN_MIB` (helper) and `JI_CACHE_MARGIN_MIB` (primary)
  are the knobs. The helper GPU holds the MTP head (needs margin 1900 today).
- **exllamav3 format drift**: assert on everything you don't handle — unknown codebook marker, `quantization_config`
  version, non-uniform codebooks, missing `svh` (`out_scales` not "always"), odd widths (handled: 2-8 bits), vocab /
  head sizes. Fail loudly with the tensor name.
- **Mixed widths per tensor** (MiniCPM-V had 3-8 bits): per-tensor ggml type `EXL3_<bits>`; pipelines exist per width.
- **TP2 hot-expert set is per model** (`ref/tp/tp_hot_*.bin`, `JI_ROUTE_DUMP` + `tools/aux-jt/routegen.py 400`;
  serve.sh picks it by model alias). A native-directory model needs the same selection (key it on the directory name
  or on `config.json`'s `_name_or_path`).
- **Never two model processes**; stop `junkinfer.service` first. **Never SIGKILL a busy TP2 process.**
- **The ggml graph path is unsafe for references** (layer-3 KV loss with big Vulkan blocks); JDE prefill is the
  reference path.
- Keep `exl3_to_gguf.py` working until native loading passes every check (it is the fallback and the oracle).
- Model files run from the SSD only.

## 5. Remaining optimisation headroom (after this plan), biggest first

| lever | today | how | expected |
|---|---|---|---|
| Sage's own MTP head (step 4) | Sage no MTP 39.6 / 45.0 / 41.7 t/s | in-model draft head | +14-23 % decode (as on Qwen3.6) |
| mul1 decode ALU (step 6) | Sage GPU 21.8 vs 18.0 ms/token | v_sad_u8 / ubyte cvt / folded constants | up to ~+20 % decode on mul1 models |
| MTP verify-step anomaly | 2-token steps cost more than 3-token steps (q8: 41.6 vs 35.8 ms, vbench 2026-10-24) | profile the nt=2 path (`vb.sh`, `JI_PROFILE_GPU`) — likely a kernel variant / workgroup split chosen badly for nt=2 | MTP gain grows (most steps are 2-token) |
| expert misses over PCIe | hit rate ~93-94 % short; misses stream from pinned host memory | bigger effective cache: KVarN-6 frees KV VRAM once its prefill bug is fixed; per-model TP hot set; `JI_TP_FRAC` tuning per model | decode at depth |
| prefill GEMM efficiency | ~45 % of peak, barrier / LDS-latency bound | double-buffered LDS in the EXL3 GEMM | prefill +20-40 % |
| sparse attention | decode 32.4 t/s at 200k | post-MVP track (`open-paths.md`: micro-blocks M1-M2 first) | long-context decode |

## 6. Effort summary

| step | effort | depends on |
|---|---|---|
| 1 reader + plan, byte-exact | 1-2 days | — |
| 2 metadata / tokenizer | 1 day | 1 |
| 3 text model native | 1 day | 1, 2 |
| 4 MTP from checkpoint | 2-3 days | 3 |
| 5 vision | 3-5 days | 3 (+ reference mmproj download) |
| 6 fast mul1 | 1-2 days | — (parallel) |

Total ~1.5-2.5 weeks of focused work. Order: 6 can start at once (independent, immediate speed); then 1 -> 2 -> 3 -> 4
(user-visible speed: MTP); 5 last (largest, needs a download and the M-RoPE work).

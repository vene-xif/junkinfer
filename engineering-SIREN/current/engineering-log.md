# Engineering log

> Chronological. Append only.

## 2026-10-15 — project start

**Survey** of Strata, BELLS, ik_llama, BeeLlama, exllamav3 (verdicts in `candidate-ledger.md`).

**AUX-1 box state was wrong** (Problem / Fix / Why / Evidence):
- Problem: `amdgpu.dpm=0` was still in `/etc/default/grub` (written 12:18, boot 12:20). Every
  2026-10-15 qwen35moe number in the prior project (y1/z1/z2/w1 sweeps, the 0.1-0.3 GB/s "copy
  wall", 5.7 t/s `--cpu-moe`) ran with the GPU at boot clocks.
- Fix: removed the parameter (backup `/etc/default/grub.bak-dpm0-20261015`), `update-grub`, reboot.
- Evidence: `/proc/cmdline` clean; `pp_dpm_sclk` 300-1340 MHz, `pp_dpm_mclk` 300-2000 MHz, level auto.
- Also found: PCIe link is 3.0 x16 (prior docs said x8); RAM is DDR4-3000 (docs said 2133).
- RAM upgraded by the user to 32 GB (4 x 8 GB).

**Re-baseline** (smoke, 1 repeat, mainline llama.cpp 8bb3cf0 Vulkan, RADV_PERFTEST=nogttspill,
Qwen3.6-35B IQ4_NL 19.3 GiB, -ngl 99 -fa 1, llama-bench -p 512 -n 32):
- `-ncmoe 40`: pp512 136.1, tg32 19.9 t/s
- `-ncmoe 29`: pp512 155.3, tg32 21.9 t/s
- host-to-device (llama-bells-selftest): 6.82 pageable / 7.93 pinned GB/s, unchanged by DPM.

**Reference run started on VENE** 22:45 (`ref/run_ref.sh`, dry-run passed first).
Bugs found in review before launch: errexit is ignored inside a subshell used as an `if`
condition (stage failures would have been invisible); a preflight check could `exit` the main
shell; a `RETURN` trap would not have stopped the BF16 server on error; `grep -q` + `pipefail`
false negative on a large curl response. All fixed before the run.

## 2026-10-15 — reference-tool findings (VENE) and MVP (AUX-1)

**Implementation noise floor** (junk-ref on Qwen3.6-35B UD-IQ4_NL, 2 wiki rows, RTX 3090):
- same path twice: KL 0.0, top-1 100% (deterministic).
- standard (fused top-k) vs graph split at the routing tensors (unfused): KL 0.0053, top-1 97.1%.
- each CUDA path vs the CPU backend: KL 0.0088 / 0.0090, top-1 97.0% / 96.2%.
- Decision: reference logits come from the standard path; routing traces from a separate pass
  (stage `routes`). Quality gates must be read against this floor: two correct implementations of
  the same quantized model differ by ~0.005-0.009 KL.
- Also: llama.cpp runs the last layer only on output tokens (`inp_out_ids`), so a routing trace
  must make every token an output.

**MVP on AUX-1** (engine fork, Vulkan, smoke, 1 run): `scripts/serve.sh` + `scripts/smoke_api.py` ALL PASS
(chat, tool call parsing, prefix cache 49.5 s -> 1.4 s). 262k context + q8_0 KV fits with `-ncmoe 34`
or 36 (decode 21.4 / 20.7 t/s at short depth).

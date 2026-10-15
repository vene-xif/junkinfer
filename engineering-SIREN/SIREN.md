# SIREN rules — engineering-SIREN (JunkInfer)

## Scope

The engineering record of JunkInfer: builds, measurements, A/B results, design decisions.

## Language

English. ASD-STE100 style: short sentences, active voice, one idea per sentence, no idioms.

## Operating rules

- Read `index.md` and `current/STATE.md` before changing anything.
- **Work iteratively.** First an MVP that works end to end. Then one feature at a time.
- **Update the docs with every commit that improves something.** STATE.md in place; the
  engineering log and the candidate ledger by appending.
- **Compute the ceiling first, then measure** (`current/physics.md`). A measurement above an
  independently derived bound is wrong, not exciting.
- **Smoke tests are short.** Stop at the first timing lines. The ballpark of decode and
  prefill speed shows in seconds. A full-length run is for milestones only.
- **Perfect invariance means a disconnected knob.** Smoke-test every new flag before sweeping it.
- **Quality gate before speed claims.** Any change to kernels, quantisation, KV format or
  attention must pass the KLD gate (`current/quality-gates.md`) before its speed is believed.
- Record build id (repo, commit, flags) and box state with every number. AUX-1 box state that
  invalidated earlier numbers: `amdgpu.dpm=0` left in grub (fixed 2026-10-15).
- No deletion without approval. Supersede, do not overwrite.

## Format

- Dated entries: Problem / Fix / Why / Evidence.
- Status vocabulary: ACTIVE · RESOLVED · REVISED (date) · REJECTED · BLOCKED · RETRACTED (date).
- Result rows carry value, units, build id, repeats and date.

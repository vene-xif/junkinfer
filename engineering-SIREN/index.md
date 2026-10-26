# Index — engineering-SIREN (JunkInfer)

> **For beginners:** this is the engineering record of JunkInfer. Start with
> `current/STATE.md`. It says where the project is and what is open.

## Read in this order

1. `SIREN.md` — rules.
1b. `../HANDOFF.md` — handoff: state, access, rules, workflow, open work (start here if you are new).
1c. `current/native-exl3-plan.md` — **the top-priority work item** (native EXL3, MTP, vision, fast mul1).
2. `current/STATE.md` — current state, targets, open items (live).
3. `current/physics.md` — the hardware and model arithmetic: what is possible on this box.
4. `current/quality-gates.md` — how quality is measured (BF16 reference, KLD).
5. `current/candidate-ledger.md` — every idea considered, with its verdict and reason.
6. `current/engineering-log.md` — chronological record.

## Machines

| name | role | address |
|---|---|---|
| AUX-1 | target box: Ryzen 5 2600, 32 GB DDR4-3000, RX 580 8 GB (8 GB ReBAR), B450 | 192.168.1.4 |
| VENE | one-time reference run: RTX 3090, 128 GB. Canonical repo: `/mnt/chonker/projects/junkinfer` | 192.168.1.2 |
| CHERRY | the agent's workstation. Do not use its GPU for this project. | 192.168.1.3 |

## Prior work this project builds on

`/mnt/off/30-work/2026-10_aux1-gfx803-inference/` (on CHERRY): the vLLM/BELLS investigation on
AUX-1. Its numbers from 2026-10-15 were taken with DPM disabled and are not comparable.

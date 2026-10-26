# AUX-1 test harness (`~/jt`)

Snapshot of AUX-1's `~/jt` scripts, refreshed 2026-10-26. The live copies stay in AUX-1 `~/jt` (the scripts use
absolute `~/jt` / `~/junkinfer` paths: run them there). Sudo calls read the password from `$SUDO_PW` here (the live
copies differ only in that).

Before any run: `sudo systemctl stop junkinfer` (the boot service holds both GPUs; two model processes OOM the box).
Afterwards: `sudo systemctl start junkinfer`.

## The scripts that matter

| script | what |
|---|---|
| `kl_v2.sh SET ROW [ENV=..]` | KL of a long row vs the full-attention v2 reference (`~/jt/<SET>_r<ROW>_v2.jref`), prefix restored from `/mnt/ssd/kvcache/jref_v2` (decode-only). `JR_STATE_DIR=<empty dir>` = prefill the prefix from scratch instead. Rows: `L_code 0` 32k, `L_code 1` 130k, `L_code 2` 261k, `L_code64 0`, `L_code98 0`. Output: last line via `klsum.py`, log `~/jt/kl.log` |
| `klsum.py` | KL mean / p99 / max / top-1 from `kl.log` |
| `spgate.sh "LABEL ENV.."...` | KL gate over several rows per config (`ROWS=` env); sparse-attention gate |
| `spfgate.sh` | sparse-prefill KL gate (each config prefills its own prefix) |
| `mkrefs_v2.sh`, `mkrefs_v2b.sh` | how the v2 references + prefix states were built (1 GiB Vulkan blocks, JDE prefill for deep rows) |
| `mtpsrv.sh` | start the server in the production config (TP2 + MTP; `NMAX=0` = no MTP; env passes through) |
| `spectest.py` | decode t/s on three prompts (rewrite / story / explain), 1024 tokens, ignore_eos |
| `dpb.py` | prefill speed at depth from saved slot KV |
| `vb.sh` | verify-step (1-4 token) GPU cost with server settings |
| `kvrcmp.py`, `kvrblk.py` | K/V error per layer (and per block of cells) between two saved states |
| `stzero.py`, `stcmp*.py`, `sthdr.py` | state-file inspection (zero rows, diffs, header) |
| `tlog.sh` | GPU thermal logger / guard (superseded by `scripts/thermal_guard.sh` + its service) |
| `restart.sh` | guarded server restart (refuses if another model process runs) |
| `job*.sh` | one experiment each, chained by waiting on the previous job's `JOBnn_DONE` marker; read them as worked examples (job33-40: KVarN MTP slowdown + 130k prefill diagnosis) |

Other scripts are one-off probes from the engineering log (dates in their header comments).

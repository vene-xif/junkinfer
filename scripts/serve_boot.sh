#!/usr/bin/env bash
# Boot-time server (systemd: scripts/junkinfer.service). Foreground, LAN-facing, Qwen-Sharp chat template (serve.sh
# default). Picks the mode from the GPUs that enumerated:
#   two RX 580s -> production config (as serve_prod.sh): JI_TP=2 tensor-parallel decode, two-card JDE prefill, MTP
#   one RX 580  -> single-card JDE (a card that hung needs a cold power cycle to come back; the server still starts)
# Any JI_* in /etc/junkinfer.env overrides these defaults (e.g. JI_TP=0 JI_MTP=0 to force single card,
# JI_API_KEY=... to require a key, JI_HOST=127.0.0.1 for local only).
set -euo pipefail
cd "$(dirname "$0")/.."
ncards=$(ls -d /sys/class/drm/card[0-9] 2>/dev/null | wc -l)
export JI_HOST=${JI_HOST:-0.0.0.0} JI_PORT=${JI_PORT:-8080}
export JI_SLOT_DIR=${JI_SLOT_DIR:-/mnt/ssd/kvcache}
# MTP default: on for Qwen3.6 (its own draft head, +14-23 % decode); off for Sage-A1.5 (default model when present):
# the converter drops Sage's EXL3 MTP layer and the Qwen3.6 head gains nothing on it (2026-10-26: rewrite / story /
# explain 34.6 / 45.6 / 42.6 t/s with it vs 38.6 / 45.1 / 41.7 without)
MTP_DEF=1
[ -z "${JI_MODEL:-}" ] && [ -f /mnt/ssd/models/Sage-A1.5-exl3-4.00bpw.gguf ] && MTP_DEF=0
case "${JI_MODEL:-}" in *Sage-A1.5*) MTP_DEF=0 ;; esac
if [ "$ncards" -ge 2 ] && [ "${JI_TP:-2}" != 0 ]; then
  export JI_TP=${JI_TP:-2} JI_JPF=${JI_JPF:-1} JI_MTP=${JI_MTP:-$MTP_DEF}
  if [ "$JI_MTP" != 0 ]; then export JI_TP_MARGIN_MIB=${JI_TP_MARGIN_MIB:-1900}; else export JI_TP_MARGIN_MIB=${JI_TP_MARGIN_MIB:-1500}; fi
  echo "serve_boot: $ncards GPUs - two-card mode (JI_TP=$JI_TP, MTP=$JI_MTP)"
else
  export JI_TP=0 JI_MTP=0
  # the remaining card enumerates as Vulkan0 when only one is present
  [ "$ncards" -lt 2 ] && export JI_GPU=${JI_GPU:-0}
  echo "serve_boot: $ncards GPU(s) - single-card mode (JI_GPU=${JI_GPU:-1})"
fi
exec scripts/serve.sh "$@"

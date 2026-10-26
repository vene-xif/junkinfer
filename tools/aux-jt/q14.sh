#!/bin/bash
# dual after the sched view-copy fix: KL single vs dual, then speed at ub 2048 / 4096
~/jt/kl.sh L_code 1
DUAL=31 UB=2048 ~/jt/kl.sh L_code 1
echo "  card0 gtt after: $(cat /sys/class/drm/card0/device/mem_info_gtt_used)"
DUAL=31 UB=4096 ~/jt/kl.sh L_code 1
echo "  card0 gtt after: $(cat /sys/class/drm/card0/device/mem_info_gtt_used)"
source <(sed -n '/^short()/p;/^pf()/,/^}/p;/^depth()/p;/^kills()/p' ~/jt/q13.sh)
export JI_IGNORE_EOS=1
for cfg in "JI_DUAL=31 JI_UBATCH=2048" "JI_DUAL=31 JI_UBATCH=4096" "JI_DUAL=32 JI_UBATCH=4096"; do
  kills; ~/jt/restart.sh $cfg JI_SLOT_DIR=/mnt/ssd/kvcache >/dev/null 2>&1
  echo "== $cfg | vram0 $(cat /sys/class/drm/card0/device/mem_info_vram_used) gtt0 $(cat /sys/class/drm/card0/device/mem_info_gtt_used)"
  echo -n "  short: "; short
  echo -n "  prefill: "; pf
  depth
  echo "  gtt0 after: $(cat /sys/class/drm/card0/device/mem_info_gtt_used)"
done
kills

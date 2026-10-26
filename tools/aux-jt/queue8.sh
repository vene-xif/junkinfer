#!/bin/bash
while ! grep -q SPSPEED_DONE ~/jt/spspeed.log 2>/dev/null; do sleep 30; done
cfgs=(); for L in 3 7 11 15 19 23 27 31 35 39; do cfgs+=("JI_SPARSE=65536 JI_SP_DENSE_LAYERS=$L"); done
ROWS="L_code:1 L_code:2" ~/jt/spgate.sh "${cfgs[@]}" > ~/jt/spgate8.log 2>&1

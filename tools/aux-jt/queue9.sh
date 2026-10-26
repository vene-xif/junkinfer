#!/bin/bash
while ! grep -q SPGATE_DONE ~/jt/spgate8.log 2>/dev/null; do sleep 30; done
ROWS="L_code98:0 L_code:1 L_code:2" ~/jt/spgate.sh "JI_JPF=1 JR_TAIL_B=1025 JI_SPARSE=0" "JI_JPF=1 JR_TAIL_B=1025 JI_SPARSE=0 JI_SPARSE_PF=0" > ~/jt/spgate9.log 2>&1

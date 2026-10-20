#!/bin/bash
OUT=~/jt/bgtest5.out; rm -f $OUT
~/jt/kl.sh L_code 1 JI_SPARSE=16384 JI_SP_BG=1 >> $OUT
~/jt/kl.sh L_code 1 JI_SPARSE=16384 JI_SP_BG=2 >> $OUT
~/jt/kl.sh L_code 0 JI_SPARSE=4096 JI_SP_BG=2 >> $OUT
~/jt/kl.sh L_code 1 JI_SPARSE=8192 JI_SP_BG=1 >> $OUT
echo DONE >> $OUT

#!/bin/bash
while ! grep -q SPGATE_DONE ~/jt/spgate3.log; do sleep 30; done
~/jt/spgate.sh "JI_TP=0 JI_SPARSE=0" "JI_TP=0 JI_SPARSE=0 JI_KVSIM=4" "JI_TP=0 JI_SPARSE=0 JI_KVSIM=3" "JI_TP=0 JI_SPARSE=0 JI_KVSIM=2" > ~/jt/spgate4.log 2>&1

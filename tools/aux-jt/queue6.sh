#!/bin/bash
while ! grep -q SPSPEED_DONE ~/jt/spspeed.log 2>/dev/null; do sleep 30; done
~/jt/spgate.sh "JI_SP_BG=1" "JI_SP_BG=1 JI_SPARSE=65536" > ~/jt/spgate6.log 2>&1

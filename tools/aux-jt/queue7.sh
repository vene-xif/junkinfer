#!/bin/bash
while ! grep -q SPGATE_DONE ~/jt/spgate6.log 2>/dev/null; do sleep 30; done
~/jt/spspeed.sh > ~/jt/spspeed.log 2>&1

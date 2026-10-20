#!/bin/bash
for row in "L_code98 0" "L_code 1" "L_code 2"; do
  for e in "JI_SP_BG=2" "JI_SP_BG=3" "JI_SP_NORM=2"; do
    [ "$row" = "L_code 1" ] && [ "$e" != "JI_SP_BG=3" ] && continue
    ~/jt/kl.sh $row $e
  done
done

#!/usr/bin/env python3
# Split a RADV_DEBUG=shaders dump into compute shaders, report the ones matching a marker: max VGPR/SGPR, length
import re, sys
txt = open(sys.argv[1]).read().split("Compute Shader\ndisasm:")
mark = sys.argv[2] if len(sys.argv) > 2 else "v_readlane_b32"
for i, sh in enumerate(txt[1:]):
    sh = sh.split("\n\n\n")[0] if False else sh
    dis = sh.split("NIR")[0]
    n = dis.count(mark)
    if n < int(sys.argv[3] if len(sys.argv) > 3 else 32): continue
    vs = [int(x) for x in re.findall(r"\bv(\d+)\b", dis)] + [int(x) for x in re.findall(r"\bv\[\d+:(\d+)\]", dis)]
    ss = [int(x) for x in re.findall(r"\bs(\d+)\b", dis)] + [int(x) for x in re.findall(r"\bs\[\d+:(\d+)\]", dis)]
    ninst = len([l for l in dis.splitlines() if l.startswith("\t") and ";" in l])
    print(f"shader {i}: {mark} x{n}  max v{max(vs)} s{max(ss)}  {ninst} instrs  "
          f"mac {dis.count('v_mac_f32')} readlane {dis.count('v_readlane')} smem {dis.count('s_buffer_load')} ds_read {dis.count('ds_read')} waitcnt {dis.count('s_waitcnt')}")

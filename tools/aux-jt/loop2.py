import re, sys
txt = open(sys.argv[1]).read().split("Compute Shader\ndisasm:")
sh = [s.split("shader: MESA_SHADER")[0] for s in txt[1:]]
sh = [s for s in sh if s.count("s_buffer_load_dwordx4") >= 8]
d = sh[int(sys.argv[2])]
L = [l.split(";")[0].strip() for l in d.splitlines()]
# print control flow summary from the first mac block, collapsing mac runs
start = [n for n, l in enumerate(L) if "v_mac" in l][0]
start = max(n for n in range(start) if re.match(r"^BB\d+:", L[n]))
out = []; macs = 0
for l in L[start:start + int(sys.argv[3])]:
    if "v_mac" in l: macs += 1; continue
    if macs: out.append(f"   [{macs} v_mac]"); macs = 0
    out.append(l)
print("\n".join(out))

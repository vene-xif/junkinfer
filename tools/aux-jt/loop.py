import re, sys
txt = open(sys.argv[1]).read().split("Compute Shader\ndisasm:")
sh = [s.split("shader: MESA_SHADER")[0] for s in txt[1:]]
sh = [s for s in sh if s.count("s_buffer_load_dwordx4") >= int(sys.argv[2])]
d = sh[int(sys.argv[3]) if len(sys.argv) > 3 else 0]
blocks = []; cur = []
for l in d.splitlines():
    if re.match(r"^BB\d+:", l): blocks.append(cur); cur = [l]
    else: cur.append(l.split(";")[0].strip())
blocks.append(cur)
for b in blocks:
    n = sum("v_mac" in y for y in b)
    if n >= 16:
        print("=====", b[0], len(b), "instrs,", n, "mac")
        print("\n".join(b[1:]))

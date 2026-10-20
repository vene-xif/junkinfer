import re, collections
lines = open("/home/vene/jt/srv.log", errors="ignore").read().split("\n")
blocks, cur = [], None
for l in lines:
    m = re.search(r"jde_prof: per token ([0-9.]+) ms", l)
    if m:
        cur = {"tot": float(m.group(1)), "k": {}}; blocks.append(cur); continue
    m = re.search(r"jde_gpu:   (.+?)\s+([0-9.]+) ms/token\s+([0-9.]+) per token", l)
    if m and cur is not None:
        cur["k"][m.group(1).strip()] = float(m.group(2))
blocks = [b for b in blocks if b["k"]][5:]
V = [b for b in blocks if any(" nt2" in k for k in b["k"])]
S = [b for b in blocks if not any(" nt" in k for k in b["k"])]
def agg(bs):
    a = collections.defaultdict(float)
    for b in bs:
        for k, ms in b["k"].items():
            a[re.sub(r" nt\d", "", k)] += ms
    return {k: v / len(bs) for k, v in a.items()}
a1, a2 = agg(S), agg(V)
print("single calls", len(S), "verify calls", len(V))
keys = sorted(set(a1) | set(a2), key=lambda k: -a2.get(k, 0))
print("%-34s %8s %8s %6s" % ("kernel", "single", "verify2", "ratio"))
for k in keys[:24]:
    print("%-34s %8.2f %8.2f %6.2f" % (k, a1.get(k, 0), a2.get(k, 0), a2.get(k, 0) / max(1e-6, a1.get(k, 0))))
print("sum %.2f %.2f" % (sum(a1.values()), sum(a2.values())))

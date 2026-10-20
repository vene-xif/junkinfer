import re, sys
L = open("/tmp/live.txt").read().splitlines()
i = int(sys.argv[1])
while not L[i].startswith("\tlive in"): i -= 1
ids = L[i].split(":")[1].split()
defs = {}
for l in L[max(0, i - 4000):i]:
    m = re.search(r"\)\s+(.*?) = ", l)
    if not m: continue
    for c, d in re.findall(r"([vs]\d+): (%\d+)", m.group(1)):
        defs[d] = (c, l.split("=", 1)[1].strip()[:60])
vs = [x for x in ids if defs.get(x, ("?",))[0].startswith("v")]
print(len(ids), "live,", len(vs), "vector")
for x in vs: print(x, defs[x])

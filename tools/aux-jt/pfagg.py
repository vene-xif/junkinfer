import re, sys, collections
for fn in sys.argv[1:]:
    agg = collections.defaultdict(float); tot = 0.0
    for l in open(fn, errors="ignore"):
        m = re.match(r"^(.*?): (\d+) x ([0-9.]+) us = ([0-9.]+) us", l)
        if not m: continue
        name = m.group(1)
        key = re.sub(r" m=\d+.*| n=.*|\(.*", "", name).strip()
        mm = re.search(r"m=(\d+) n=(\d+) k=(\d+)", name)
        if mm and int(mm.group(2)) < 64: key += " (small n)"
        agg[key] += float(m.group(4)); tot += float(m.group(4))
    print("==", fn, "total %.1f s" % (tot / 1e6))
    for k, v in sorted(agg.items(), key=lambda x: -x[1])[:22]:
        print("  %-60s %7.2f s %5.1f %%" % (k[:60], v / 1e6, 100 * v / tot))

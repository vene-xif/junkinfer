import struct, collections, sys
def jtok(path):
    f = open(path, "rb"); assert f.read(4) == b"JTOK"; ver, n = struct.unpack("<II", f.read(8)); rows = []
    for _ in range(n):
        ln, a, b = struct.unpack("<III", f.read(12)); rows.append(list(struct.unpack("<%di" % ln, f.read(4 * ln))))
    return rows
D = "/home/vene/junkinfer/data/corpus/"
sets = {k: [t for r in jtok(D + k + ".jtok") for t in r] for k in ("L_code", "L_prose", "S1_wiki", "S2_code")}
for k, v in sets.items(): print(k, len(v), "tokens", len(set(v)), "distinct")
def cov(train, test, K):
    c = collections.Counter(train); keep = {t for t, _ in c.most_common(K)} | set(range(248044, 248320))   # + special tokens
    return sum(1 for t in test if t in keep) / len(test)
for test in ("L_prose", "L_code"):
    train = [t for k, v in sets.items() if k != test for t in v]
    print("held out %-8s" % test, " ".join("K=%dk %.3f" % (K // 1024, cov(train, sets[test], K)) for K in (16384, 32768, 40960, 65536)))

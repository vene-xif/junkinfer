import struct
def jtok(path):
    f = open(path, "rb"); assert f.read(4) == b"JTOK"; ver, n = struct.unpack("<II", f.read(8)); rows = []
    for _ in range(n):
        ln, a, b = struct.unpack("<III", f.read(12)); rows.append(list(struct.unpack("<%di" % ln, f.read(4 * ln))))
    return rows
D = "/home/vene/junkinfer/data/corpus/"
for k in ("L_code", "L_prose", "S1_wiki", "S2_code"):
    v = [t for r in jtok(D + k + ".jtok") for t in r]
    print(k, len(v), " ".join("id<%dk %.3f" % (K // 1024, sum(1 for t in v if t < K or t >= 248044) / len(v)) for K in (16384, 32768, 49152, 65536, 98304)))

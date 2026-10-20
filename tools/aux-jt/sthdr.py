import json, struct, os, sys, collections
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
hdr = {}
for fn in sorted(set(idx.values())):
    with open(os.path.join(d, fn), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    for k, v in h.items():
        if k != "__metadata__": hdr[k] = v
print("tensors", len(hdr))
pre = collections.Counter(".".join(k.split(".")[:2]) for k in hdr)
print("prefixes", pre.most_common(8))
suff = collections.Counter(k.split(".")[-1] for k in hdr)
print("suffixes", suff.most_common(20))
bits = collections.Counter()
for k, v in hdr.items():
    if k.endswith(".trellis"):
        grp = "experts" if ".experts." in k else ("vision" if "vpm" in k or "vision" in k else k.split(".")[-2])
        bits[(grp, v["shape"][-1] // 16)] += 1
for kv in sorted(bits.items()): print(kv)
for k in list(hdr)[:3]: print(k, hdr[k])
for k in hdr:
    if "embed_tokens" in k or "lm_head" in k: print(k, hdr[k]["dtype"], hdr[k]["shape"])
ex = [k for k in hdr if k.endswith((".mul1", ".mcg"))][:2]
for k in ex: print(k, hdr[k])

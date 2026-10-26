import numpy as np, sys
N=31743; KV=262144
def load(p):
    a=np.fromfile(p,dtype=np.uint8).reshape(2,KV,272)[:, :N]
    b=a.reshape(2,N,8,34)
    d=b[...,:2].copy().view(np.float16).astype(np.float32)   # [2,N,8,1]
    q=b[...,2:].copy().view(np.int8).astype(np.float32)      # [2,N,8,32]
    return (q*d).reshape(2,N,256)
for L in sys.argv[1:]:
    for kv in "kv":
        g=load(f"/mnt/ssd/kvd_mpx_{L}.{kv}"); j=load(f"/mnt/ssd/kvd_t2_{L}.{kv}")
        err=np.linalg.norm(g-j,axis=2)/np.maximum(np.linalg.norm(g,axis=2),1e-9)   # [2,N]
        per=err.mean(0)
        blk=[per[i:i+4096].mean() for i in range(0,N,4096)]
        print(f"L{L} {kv}: rel err mean {per.mean():.4f} median {np.median(per):.4f} max {per.max():.3f} | per 4096 block", " ".join(f"{x:.4f}" for x in blk))
        # within-block position pattern
        pos=np.array([per[i::4096][:7].mean() for i in range(0,4096,512)])
        print("   by offset in ubatch (every 512):", " ".join(f"{x:.4f}" for x in pos))
        worst=np.argsort(per)[-5:]; print("   worst cells", worst.tolist(), per[worst].round(3).tolist())

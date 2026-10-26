import numpy as np, sys
N=31743
A=np.memmap(sys.argv[1],dtype=np.uint8,mode="r"); B=np.memmap(sys.argv[2],dtype=np.uint8,mode="r")
mk=np.frombuffer(bytes([8,0,0,0,0x20,2,0,0,0,0,0,0]),dtype=np.uint8)
raw=bytes(A[:300<<20])
offs=[]; i=raw.find(bytes(mk))
while i>=0: offs.append(i); i=raw.find(bytes(mk),i+1)
print("markers",len(offs), offs[:3])
def deq(X,o):
    a=np.array(X[o+12:o+12+N*544]).reshape(N,2,8,34)
    d=a[...,:2].copy().view(np.float16).astype(np.float32); q=a[...,2:].copy().view(np.int8).astype(np.float32)
    return (q*d).reshape(N,2,256)
attn=[3,7,11,15,19,23,27,31,35,39]
for n,o in enumerate(offs[:20]):
    g=deq(A,o); j=deq(B,o)
    e=np.linalg.norm(g-j,axis=2)/np.maximum(np.linalg.norm(g,axis=2),1e-9)
    per=e.mean(1)
    blk=" ".join(f"{per[i:i+4096].mean():.3f}" for i in range(0,N,4096))
    print(f"{"KV"[n//10]} L{attn[n%10]:2d}: rel err mean {per.mean():.4f} med {np.median(per):.4f} | per 4096: {blk} | first 8 cells {per[:8].round(3).tolist()}")

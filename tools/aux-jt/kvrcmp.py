import numpy as np, sys
# per-layer K/V relative error between two slot files (same cell count); rows of 544 B (2 heads x 272 B q8_0)
A=np.memmap(sys.argv[1],dtype=np.uint8,mode="r"); B=np.memmap(sys.argv[2],dtype=np.uint8,mode="r")
mk=bytes([8,0,0,0,0x20,2,0,0,0,0,0,0]); raw=bytes(A[:])
offs=[]; i=raw.find(mk)
while i>=0: offs.append(i); i=raw.find(mk,i+1)
N=(offs[1]-offs[0]-12)//544
def deq(X,o):
    a=np.array(X[o+12:o+12+N*544]).reshape(N,2,8,34)
    d=a[...,:2].copy().view(np.float16).astype(np.float32); q=a[...,2:].copy().view(np.int8).astype(np.float32)
    return (q*d).reshape(N,2,256)
print("cells",N,"markers",len(offs))
for n,o in enumerate(offs):
    g=deq(A,o); j=deq(B,o)
    e=np.linalg.norm(g-j,axis=2)/np.maximum(np.linalg.norm(g,axis=2),1e-9)
    per=e.mean(1)
    print(f"{"KV"[n//10]} {n%10}: rel err mean {per.mean():.4f} | first 8192 {per[:8192].mean():.4f} | sealed part {per[8192:-256].mean():.4f} | last 256 {per[-256:].mean():.4f} | worst cell {int(per.argmax())} {per.max():.3f}")

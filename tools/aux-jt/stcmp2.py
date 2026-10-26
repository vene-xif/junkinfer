import numpy as np, sys
N=31743
A=np.memmap(sys.argv[1],dtype=np.uint8,mode="r"); B=np.memmap(sys.argv[2],dtype=np.uint8,mode="r")
mk=bytes([8,0,0,0,0x20,2,0,0,0,0,0,0]); raw=bytes(A[:300<<20])
offs=[]; i=raw.find(mk)
while i>=0: offs.append(i); i=raw.find(mk,i+1)
def deq(X,o):
    a=np.array(X[o+12:o+12+N*544]).reshape(N,2,8,34)
    d=a[...,:2].copy().view(np.float16).astype(np.float32); q=a[...,2:].copy().view(np.int8).astype(np.float32)
    return (q*d).reshape(N,2,256)
for n in [0,1,9]:
    o=offs[n]; g=deq(A,o); j=deq(B,o)
    e=(np.linalg.norm(g-j,axis=2)/np.maximum(np.linalg.norm(g,axis=2),1e-9))
    for h in range(2):
        bad=e[:,h]>0.2
        # runs
        runs=[]; s=None
        for c in range(N):
            if bad[c] and s is None: s=c
            if not bad[c] and s is not None: runs.append((s,c-1)); s=None
        if s is not None: runs.append((s,N-1))
        big=[r for r in runs if r[1]-r[0]>16]
        print(f"marker {n} head {h}: bad cells {bad.sum()} runs>16: {big[:12]}  nruns {len(runs)}")
    jn=np.linalg.norm(j,axis=2); print("   JPF norms at 8190..8194 head0", jn[8190:8195,0].round(2).tolist(), " graph", np.linalg.norm(g,axis=2)[8190:8195,0].round(2).tolist())

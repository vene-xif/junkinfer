import numpy as np, sys
# per-layer K/V relative error per block of cells between two slot files (same cells): where does a state go wrong
A=np.memmap(sys.argv[1],dtype=np.uint8,mode="r"); B=np.memmap(sys.argv[2],dtype=np.uint8,mode="r")
BL=int(sys.argv[3]) if len(sys.argv)>3 else 8192
mk=bytes([8,0,0,0,0x20,2,0,0,0,0,0,0])
def find_all(X):
    raw=bytes(X[:min(len(X),1<<31)]) if len(X)<(1<<31) else None
    offs=[]; 
    step=1<<26; pos=0
    while pos<len(X):
        chunk=bytes(X[pos:min(len(X),pos+step+12)]); i=chunk.find(mk)
        while i>=0:
            if i<step and (not offs or offs[-1]!=pos+i): offs.append(pos+i)
            i=chunk.find(mk,i+1)
        pos+=step
    return offs
oa=find_all(A); ob=find_all(B)
N=(oa[1]-oa[0]-12)//544; Nb=(ob[1]-ob[0]-12)//544
print("cells",N,Nb,"markers",len(oa),len(ob)); N=min(N,Nb)
def deq(X,o,lo,hi):
    a=np.array(X[o+12+lo*544:o+12+hi*544]).reshape(hi-lo,2,8,34)
    d=a[...,:2].copy().view(np.float16).astype(np.float32); q=a[...,2:].copy().view(np.int8).astype(np.float32)
    return (q*d).reshape(hi-lo,2,256)
for n in range(min(len(oa),len(ob))):
    row=[]
    for lo in range(0,N,BL):
        hi=min(N,lo+BL); g=deq(A,oa[n],lo,hi); j=deq(B,ob[n],lo,hi)
        e=np.linalg.norm(g-j,axis=2)/np.maximum(np.linalg.norm(j,axis=2),1e-9); row.append(e.mean())
    print(f"{"KV"[n//10]}{n%10}: "+" ".join(f"{x:.3f}" for x in row), flush=True)

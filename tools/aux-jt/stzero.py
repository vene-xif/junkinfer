import numpy as np, sys
A=np.memmap(sys.argv[1],dtype=np.uint8,mode="r"); raw=bytes(A[:])
mk=bytes([8,0,0,0,0x20,2,0,0,0,0,0,0])
offs=[]; i=raw.find(mk)
while i>=0: offs.append(i); i=raw.find(mk,i+1)
N=(offs[1]-offs[0]-12)//544 if len(offs)>1 else 0
out=[]
for o in offs:
    a=np.frombuffer(raw[o+12:o+12+N*544],dtype=np.uint8).reshape(-1,544)
    z=~a.any(axis=1); out.append(f"{z.sum()}@{np.argmax(z) if z.any() else -1}")
print(sys.argv[1].split("/")[-1], "cells~",N, len(offs), " ".join(out))

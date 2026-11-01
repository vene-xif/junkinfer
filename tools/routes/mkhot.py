# routing counts from traces -> JI_TP_HOT file (u32 [n_layer * n_expert], layer-major, as JI_ROUTE_DUMP writes)
# usage (VENE, data dir): mkhot.py OUT.bin TRACE... (traces of the served model; recomputes ids of pre-fix TP2 traces from x)
import numpy as np, sys, glob
sys.path.insert(0, '.')
from sim_cache import load_raw, router_topk
R = np.load(__import__('os').environ.get('JI_ROUTER_NPZ', 'router/router.npz')); gate = R['gate'].astype(np.float32); nw = R['norm'].astype(np.float32); eps = float(R['eps'])
out, files = sys.argv[1], sys.argv[2:]
cnt = None
for f in files:
    ids, X, keys, (nl, ne, nu, nemb) = load_raw([f])
    T = len(ids)
    if (ids == 255).any():
        for L in range(nl):
            for c in range(0, T, 4096):
                ids[c:c+4096, L] = router_topk(gate[L], nw[L], eps, X[c:c+4096, L].astype(np.float32), nu, True)[0]
    if cnt is None: cnt = np.zeros(nl * ne, np.uint64)
    np.add.at(cnt, (ids + (np.arange(nl) * ne)[None, :, None]).ravel(), 1)
    print(f, T)
cnt.astype(np.uint32).tofile(out)
print("->", out, int(cnt.sum()))

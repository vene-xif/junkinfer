#!/usr/bin/env python3
"""Expert-cache hit-rate simulator over JRTR routing traces.

Every (token, MoE layer) routes to k experts. A cache holds N experts per layer on the GPU.
Policies:
  static-self   top-N by frequency on the evaluated trace itself (optimistic upper bound for static)
  static-cross  top-N by frequency on a DIFFERENT trace (a shipped profile)
  lru           least-recently-used, per layer
  lfu-decay     per-layer usage counts with exponential decay; swap in at most S experts every R tokens
  seed+lru      static-cross seed for half the slots, LRU over the other half
Output: hit rate = fraction of the token's routed experts that are resident.
usage: cachesim.py eval.jrtr profile.jrtr [N ...]
"""
import struct, sys
from collections import OrderedDict
import numpy as np


def load(path):
    rows = []
    with open(path, "rb") as f:
        assert f.read(4) == b"JRTR"
        ver, nl, k = struct.unpack("<III", f.read(12))
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            r, L = struct.unpack("<II", h)
            ids = np.frombuffer(f.read(L * nl * k), dtype=np.uint8).reshape(L, nl, k)
            f.read(L * nl * k * 2)
            rows.append(ids)
    ids = np.concatenate(rows, 0)
    layers = [l for l in range(nl) if (ids[:, l] != 255).all()]
    return ids[:, layers, :]


def freq(ids):
    T, NL, K = ids.shape
    c = np.zeros((NL, 256), np.int64)
    for l in range(NL):
        c[l] = np.bincount(ids[:, l, :].ravel(), minlength=256)
    return c


def static(ids, prof, N):
    resident = np.argsort(-prof, axis=1)[:, :N]
    T, NL, K = ids.shape
    hit = 0
    for l in range(NL):
        m = np.zeros(256, bool); m[resident[l]] = True
        hit += m[ids[:, l, :]].sum()
    return hit / ids.size


def lru(ids, N, seed=None, seed_n=0):
    T, NL, K = ids.shape
    hit = 0
    for l in range(NL):
        pinned = set(int(x) for x in seed[l][:seed_n]) if seed is not None else set()
        last = OrderedDict()  # LRU part only
        cap = N - len(pinned)
        col = ids[:, l, :].tolist()
        for t in range(T):
            for e in col[t]:
                if e in pinned:
                    hit += 1
                elif e in last:
                    hit += 1; last.move_to_end(e)
                elif cap > 0:
                    if len(last) >= cap:
                        last.popitem(last=False)
                    last[e] = True
    return hit / ids.size


def lfu_decay(ids, N, prof, R=4, S=8, decay=0.98):
    T, NL, K = ids.shape
    hit = 0
    for l in range(NL):
        res = set(np.argsort(-prof[l])[:N].tolist())
        score = prof[l].astype(np.float64) / max(prof[l].sum(), 1) * 50
        col = ids[:, l, :]
        for t in range(T):
            for e in col[t]:
                hit += int(e) in res
            score *= decay
            score[col[t]] += 1.0
            if t % R == R - 1:
                order = np.argsort(-score)
                want = [int(e) for e in order[:N]]
                missing = [e for e in want if e not in res][:S]
                if missing:
                    evict = sorted(res, key=lambda e: score[e])[:len(missing)]
                    for a, b in zip(evict, missing):
                        res.discard(a); res.add(b)
    return hit / ids.size


def main():
    ev, pr = load(sys.argv[1]), load(sys.argv[2])
    Ns = [int(x) for x in sys.argv[3:]] or [32, 48, 64, 96, 128]
    fe, fp = freq(ev), freq(pr)
    seed = np.argsort(-fp, axis=1)
    print(f"eval {sys.argv[1]}: {ev.shape[0]} tokens x {ev.shape[1]} MoE layers x top-{ev.shape[2]}")
    print(f"{'N/layer':>8} {'%cached':>8} {'static-self':>12} {'static-cross':>13} {'lru':>6} {'seed+lru':>9} {'lfu-decay':>10}")
    for N in Ns:
        print(f"{N:>8} {100 * N / 256:>7.1f}% {static(ev, fe, N):>12.3f} {static(ev, fp, N):>13.3f} "
              f"{lru(ev, N):>6.3f} {lru(ev, N, seed, N // 2):>9.3f} {lfu_decay(ev, N, fp):>10.3f}", flush=True)


if __name__ == "__main__":
    main()

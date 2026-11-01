#!/usr/bin/env python3
"""Offline expert-cache simulator over JDE routing traces (JI_ROUTE_LOG, see read_trace.py).

Models the TP2 production split: the helper GPU holds a static hot set (JI_TP_HOT counts, first N_HELPER experts by
count), the hottest of them (JI_TP_DUP share of the traffic) are pinned on both GPUs; the primary caches the rest in
C slots. Only primary-owned accesses can miss. Policies:
  cur     - the engine's policy (jde_cache_update): score decay, +1 per access, after each token fill free slots
            (<= 64) / evict <= 8 lowest-scored residents with hysteresis; fills usable from the next token
  lru     - demand LRU, every miss is inserted (unlimited swaps)
  opt     - Belady (furthest next use), demand fill: the floor on transferred experts for C slots
  la<D>   - cur + router lookahead: at layer L (MoE input x_L known) the router of layer L+D on norm(x_L) predicts
            top-K; up to B missing primary experts are prefetched (evicting the lowest scores), usable at L+D
  orD     - same with the true ids of layer L+D (oracle predictor, same budget): the predictor ceiling
Also reports router reproduction (layer L router on x_L vs logged ids) and lookahead recall@K.

usage: sim_cache.py ROUTER.npz HOT.bin TRACE... [--C 2984] [--K 8] [--B 4] [--max-tok N]
"""
import argparse, glob, heapq, struct, sys, time
from collections import OrderedDict
import numpy as np

TOKN = 0x4e4b4f54


def load_raw(paths, max_tok=None):
    """every record in file order (MTP drafts and their re-logs included: that is the real access stream)"""
    ids_l, x_l, key_l = [], [], []
    for p in paths:
        with open(p, "rb") as f:
            hdr = struct.unpack("<8I", f.read(32))
            nl, ne, nu, nemb = hdr[2], hdr[3], hdr[4], hdr[5]
            data = f.read()
        per = nu * 3 + 2 * nemb
        rec = 16 + per * nl
        off = 0
        while off + 16 <= len(data):
            tag, pos, tok, gen = struct.unpack_from("<4i", data, off)
            hx = (tag & 0xffffffff) == TOKN
            n = 16 + (per if hx else nu * 3) * nl
            if off + n > len(data):
                break
            if hx:
                a = np.frombuffer(data, np.uint8, per * nl, off + 16).reshape(nl, per)
                ids_l.append(a[:, :nu])
                x_l.append(a[:, 3 * nu:].copy().view(np.float16))
                key_l.append((p, gen, pos))
            off += n
            if max_tok and len(ids_l) >= max_tok:
                break
        if max_tok and len(ids_l) >= max_tok:
            break
    return np.stack(ids_l).astype(np.int32), np.stack(x_l), key_l, (nl, ne, nu, nemb)


def router_topk(gate, normw, eps, x, k, one_plus):
    """x [N, E] f32 -> top-k ids [N, k] of softmax(gate @ rmsnorm(x) * w)"""
    xn = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    xn = xn * ((1.0 + normw) if one_plus else normw)
    lg = xn @ gate.T
    return np.argpartition(-lg, k - 1, axis=-1)[:, :k], lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("router"); ap.add_argument("hot"); ap.add_argument("traces", nargs="+")
    ap.add_argument("--C", type=int, default=2984); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--B", type=int, default=4); ap.add_argument("--n-helper", type=int, default=2269)
    ap.add_argument("--dup", type=float, default=0.25); ap.add_argument("--max-tok", type=int, default=0)
    ap.add_argument("--policies", default="cur,lru,opt,la1,la2,or1,or2")
    a = ap.parse_args()
    t0 = time.time()
    R = np.load(a.router)
    gate, normw, eps = R["gate"].astype(np.float32), R["norm"].astype(np.float32), float(R["eps"])
    ids, X, keys, (nl, ne, nu, nemb) = load_raw(a.traces, a.max_tok or None)
    T = len(ids)
    print(f"{T} records from {len(a.traces)} trace(s), layers {nl}, experts {ne}, top-{nu}  ({time.time() - t0:.1f} s)")

    # traces before the 2026-10-31 router_fused fix lack the TP2 helper's selections (logged as 255): recompute every
    # layer's top-k from the logged MoE input (exact up to f16 rounding of x; checked against the valid logged ids)
    if (ids == 255).any():
        R1 = np.empty_like(ids)
        for L in range(nl):
            for c in range(0, T, 4096):
                R1[c:c + 4096, L] = router_topk(gate[L], normw[L], eps, X[c:c + 4096, L].astype(np.float32), nu, True)[0]
        v = ids != 255
        agree = (R1[:, :, :, None] == ids[:, :, None, :]).any(2)[v].mean()
        print(f"{np.mean(~v):.3f} of the logged ids missing (TP2 helper share): recomputed from x, agreement on the rest {agree:.4f}")
        ids = R1

    # ---- static TP2 split (as jde.cpp TP init with the recorded counts)
    cnt = np.fromfile(a.hot, np.uint32).astype(np.float64)
    order = np.argsort(-cnt, kind="stable")
    owner = np.zeros(nl * ne, np.int8)            # 0 primary, 1 helper, 2 both (pinned)
    owner[order[:a.n_helper]] = 1
    acc = np.cumsum(cnt[order[:a.n_helper]])
    owner[order[:a.n_helper][acc - cnt[order[:a.n_helper]] < a.dup * cnt.sum()]] = 2
    keyarr = ids + (np.arange(nl) * ne)[None, :, None]    # [T, L, k] global keys
    own = owner[keyarr]
    print(f"TP2 split: helper {int((owner == 1).sum())} + pinned {int((owner == 2).sum())}; trace selections: "
          f"helper {np.mean(own == 1):.3f}, pinned {np.mean(own == 2):.3f}, primary-cached {np.mean(own == 0):.3f}")

    # ---- router reproduction and lookahead recall
    Xf = X.astype(np.float32)
    sub = np.arange(0, T, max(1, T // 1500))
    for one_plus in (True, False):
        m = []
        for L in range(nl):
            p, _ = router_topk(gate[L], normw[L], eps, Xf[sub, L], nu, one_plus)
            m.append(np.mean([len(set(p[i]) & set(ids[s, L])) for i, s in enumerate(sub)]) / nu)
        print(f"router reproduction (norm {'1+w' if one_plus else 'w'}): mean overlap {np.mean(m):.4f}")
        if np.mean(m) > 0.9:
            OP = one_plus
            break
    else:
        sys.exit("router reproduction failed: wrong x position or norm")
    pred = {}   # D -> [T, L, K] predicted ids for layer L+D made at layer L (L+D < nl)
    for D in (1, 2, 3):
        P = np.full((T, nl, a.K), -1, np.int32)
        rec = []
        for L in range(nl - D):
            p, _ = router_topk(gate[L + D], normw[L + D], eps, Xf[:, L], a.K, OP)
            P[:, L] = p
            tgt = ids[:, L + D]
            rec.append(np.mean((p[:, :, None] == tgt[:, None, :]).any(1).sum(1) / nu))
        pred[D] = P
        print(f"lookahead D={D}: recall of the true top-{nu} in predicted top-{a.K}: {np.mean(rec):.3f}  "
              f"(by layer: {' '.join(f'{r:.2f}' for r in rec[::5])})")
    # previous-token baseline: the same layer's ids of the previous record
    prev = np.mean((ids[1:, :, :, None] == ids[:-1, :, None, :]).any(-1).mean(-1))
    print(f"previous-token reuse (same layer): {prev:.3f}")
    print(f"({time.time() - t0:.1f} s)")

    # ---- cache simulations over the primary stream
    pol = a.policies.split(",")
    res = {}
    acc_keys = [[keyarr[t, L][own[t, L] == 0] for L in range(nl)] for t in range(T)]
    for p in [q for q in pol if q.startswith("cur") and q != "cur"]:   # cur<decay>e<evict>[h<hyst>], e.g. cur0.95e16
        body, _, hy = p[3:].partition("h")
        dec, _, ev = body.partition("e")
        res[p] = sim_cur(T, nl, ne, a.C, acc_keys, owner, keyarr, own, None, 0, 0, decay=float(dec or 0.98),
                         swaps_evict=int(ev or 8), hyst=float(hy or 1.25))
    if "cur" in pol or any(p.startswith(("la", "or")) for p in pol):
        for p in [q for q in pol if q == "cur" or q.startswith(("la", "or"))]:
            D = int(p[2:]) if p != "cur" else 0
            res[p] = sim_cur(T, nl, ne, a.C, acc_keys, owner, keyarr, own,
                             None if D == 0 else (pred[D] if p.startswith("la") else None), D, a.B, oracle=p.startswith("or"), ids=ids)
    for p in [q for q in pol if q.startswith("wt")]:   # wt<decay>[p<pool>][la<D>|or<D>][f<N>], e.g. wt0.98p64la1
        body, la = p[2:], None
        fut = 0
        if "f" in body:   # f<N>: victims prefer experts not used in the next N records (perfect N-token lookahead)
            body, fs = body.split("f"); fut = int(fs)
        for tag in ("la", "or"):
            if tag in body:
                body, D = body.split(tag); la = (tag, int(D))
        dec, _, pool = body.partition("p")
        P = None
        if la:
            P = pred[la[1]] if la[0] == "la" else "oracle"
        res[p] = sim_wt(T, nl, ne, a.C, acc_keys, owner, float(dec or 0.98), int(pool or 64),
                        pred=P, D=la[1] if la else 0, B=a.B, keyarr=keyarr, own=own, fut=fut)
    if "lru" in pol:
        res["lru"] = sim_lru(acc_keys, a.C)
    if "opt" in pol:
        res["opt"] = sim_opt(acc_keys, a.C)
    warm = min(500, T // 4)
    print(f"\nprimary misses per record (after {warm} warm-up records), C={a.C}, prefetch budget B={a.B}/layer, K={a.K}:")
    for p, r in res.items():
        miss = r["miss"][warm:].mean()
        line = f"  {p:4s} misses/tok {miss:6.2f}"
        if "fill" in r:
            line += f"  fills/tok {r['fill'][warm:].mean():6.2f}"
        if "pf" in r:
            line += f"  prefetched/tok {r['pf'][warm:].mean():6.2f} (used {r['pfuse'][warm:].mean():5.2f})"
        print(line)
    print(f"({time.time() - t0:.1f} s)")


def sim_cur(T, nl, ne, C, acc_keys, owner, keyarr, own, pred, D, B, oracle=False, ids=None,
            decay=0.98, hyst=1.25, swaps_fill=64, swaps_evict=8):
    N = nl * ne
    score = np.zeros(N, np.float32)
    cached = np.zeros(N, bool)
    prim = owner == 0
    # seed (JI_CACHE_SEED): the C most-recorded primary experts are resident at the start
    miss = np.zeros(T); fill = np.zeros(T); pf = np.zeros(T); pfuse = np.zeros(T)
    pfflag = np.zeros(N, bool)
    nres = 0
    for t in range(T):
        for L in range(nl):
            if pred is not None or oracle:
                if L + D < nl and B > 0:
                    if oracle:
                        cand = keyarr[t, L + D][own[t, L + D] == 0]
                    else:
                        cand = pred[t, L] + (L + D) * ne
                        cand = cand[prim[cand]]
                    cand = cand[~cached[cand]][:B]
                    for k in cand:
                        if nres >= C:
                            rk = np.flatnonzero(cached)
                            v = rk[np.argmin(score[rk])]
                            cached[v] = False; pfflag[v] = False; nres -= 1
                        cached[k] = True; pfflag[k] = True; nres += 1
                        score[k] = max(score[k], 0.51)
                        pf[t] += 1
            ks = acc_keys[t][L]
            h = cached[ks]
            miss[t] += (~h).sum()
            pfuse[t] += pfflag[ks].sum()
            pfflag[ks] = False
            np.add.at(score, ks, 1.0)
        # end of token: decay, then the engine's swap plan (usable from the next token)
        score *= decay
        want = np.flatnonzero(~cached & prim & (score > 0.5))
        if len(want) == 0:
            continue
        want = want[np.argsort(-score[want])][:swaps_fill + swaps_evict]
        free = C - nres
        rk = np.flatnonzero(cached)
        low = rk[np.argsort(score[rk])[:swaps_evict]] if len(rk) else rk
        nf = nev = ir = 0
        for k in want:
            if free > 0 and nf < swaps_fill:
                free -= 1; nf += 1
            elif nev < swaps_evict and ir < len(low) and score[k] > score[low[ir]] * hyst + 0.5:
                cached[low[ir]] = False; pfflag[low[ir]] = False; ir += 1; nev += 1; nres -= 1
            else:
                break
            cached[k] = True; nres += 1
            fill[t] += 1
    return {"miss": miss, "fill": fill, "pf": pf, "pfuse": pfuse}


def sim_wt(T, nl, ne, C, acc_keys, owner, decay, pool, pred=None, D=0, B=0, keyarr=None, own=None, fut=0):
    """write-through fill: the zero-copy miss kernel also stores the expert into a victim slot (no extra PCIe
    traffic). Victims: the `pool` lowest-scored residents picked at the start of each token (the GPU takes them in
    order, misses beyond the pool stay plain zero-copy); scores as the engine's (decay, +1 per access)."""
    N = nl * ne
    score = np.zeros(N, np.float32)
    cached = np.zeros(N, bool)
    miss = np.zeros(T); pf = np.zeros(T); pfuse = np.zeros(T)
    pfflag = np.zeros(N, bool)
    prim = owner == 0
    nres = 0
    nxt = None
    if fut:   # next record index each key is used at (for the lookahead oracle)
        nxt_use = np.full(N, 1 << 30, np.int64); nxt = np.empty(T, object)
        for t in range(T - 1, -1, -1):
            nxt[t] = nxt_use.copy() if t % 1 == 0 else None
            for ks in acc_keys[t]: nxt_use[ks] = t
    for t in range(T):
        rk = np.flatnonzero(cached)
        free = C - nres
        if len(rk) and pool > free:
            q = min(pool - free, len(rk))
            sc = score[rk].copy()
            if fut:   # used within the next fut records: never a victim before the unused ones
                sc = sc + (nxt[t][rk] <= t + fut) * 1e6
            vic = list(rk[np.argpartition(sc, q - 1)[:q]]) if q < len(rk) else list(rk)
            vic.sort(key=lambda k: sc[np.searchsorted(rk, k)])
        else:
            vic = []
        vi = 0
        for L in range(nl):
            if pred is not None and L + D < nl:   # prefetch (async DMA) into victims, usable at layer L + D
                if isinstance(pred, str):
                    cand = keyarr[t, L + D][own[t, L + D] == 0]
                else:
                    cand = pred[t, L] + (L + D) * ne
                    cand = cand[prim[cand]]
                for k in cand[~cached[cand]][:B].tolist():
                    if free > 0:
                        free -= 1; nres += 1
                    elif vi < len(vic):
                        v = vic[vi]; vi += 1
                        if not cached[v]:
                            continue
                        cached[v] = False; pfflag[v] = False
                    else:
                        break
                    cached[k] = True; pfflag[k] = True; pf[t] += 1
            ks = acc_keys[t][L]
            pfuse[t] += pfflag[ks].sum(); pfflag[ks] = False
            for k in ks[~cached[ks]].tolist():
                miss[t] += 1
                if free > 0:
                    free -= 1; nres += 1; cached[k] = True
                elif vi < len(vic):
                    v = vic[vi]; vi += 1
                    if cached[v] and v not in ks:
                        cached[v] = False; cached[k] = True
            np.add.at(score, ks, 1.0)
        score *= decay
    return {"miss": miss, "fill": np.zeros(T), "pf": pf, "pfuse": pfuse}


def sim_lru(acc_keys, C):
    T = len(acc_keys)
    od = OrderedDict(); miss = np.zeros(T)
    for t in range(T):
        for ks in acc_keys[t]:
            for k in ks.tolist():
                if k in od:
                    od.move_to_end(k)
                else:
                    miss[t] += 1; od[k] = 1
                    if len(od) > C:
                        od.popitem(last=False)
    return {"miss": miss, "fill": miss.copy()}


def sim_opt(acc_keys, C):
    seq = [(t, k) for t in range(len(acc_keys)) for ks in acc_keys[t] for k in ks.tolist()]
    n = len(seq)
    nxt = [n] * n; last = {}
    for i in range(n - 1, -1, -1):
        k = seq[i][1]; nxt[i] = last.get(k, n); last[k] = i
    cur = {}   # key -> next use
    heap = []  # (-next_use, key)
    miss = np.zeros(len(acc_keys))
    for i, (t, k) in enumerate(seq):
        if k not in cur:
            miss[t] += 1
            if len(cur) >= C:
                while True:
                    nu_, v = heapq.heappop(heap)
                    if cur.get(v) == -nu_:
                        del cur[v]; break
        cur[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return {"miss": miss, "fill": miss.copy()}


if __name__ == "__main__":
    main()

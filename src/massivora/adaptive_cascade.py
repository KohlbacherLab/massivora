"""
Self-consistent GaussDCA -> plmDCA hand-off threshold (`couple.cascade.adaptive`).

The gated population (Neff/L >= gate) is processed in batches in descending
GaussDCA rank (or by expected hits per GPU-second, `order: cost`). Each batch
also spends `explore_fraction` of its pairs on random unseen pairs below the
frontier, stratified over equal-count rank bins, so the plmDCA hit rate p(r)
is observed at every rank, not only above the frontier.

Estimator: per rank bin a Beta posterior of the hit rate (prior mean = pooled
observed rate, strength `prior_strength`), drawn `draws` times; every draw is
made monotone non-increasing in rank (weighted pool-adjacent-violators), so a
bin with few samples borrows from its neighbours and a deeper bin never gets a
higher rate than a shallower one. The unseen pairs of each bin times its rate
give the hits still missing.

Threshold: t* = the fewest further pairs g (in the current processing order)
such that the `ucb` quantile of the missed hits after processing them is
<= (1 - target_recall) x the expected total hits (posterior mean). The planned
total T = processed + g*. Stop when g* = 0 (the frontier has passed t*) and T
moved by at most one batch over the last `stable_batches` batches.

Hits: `detector` (plmDCA high_precision_hit) or `top_fraction` (plmDCA
log_lr_raw >= the (1 - q) quantile of the gated population, the quantile
estimated each batch from the processed pairs, Horvitz-Thompson weighted: the
exploration pairs of a bin stand in for its unseen pairs).

Pure numpy; the simulator (perf/day/adaptive) and the DB driver
(`massivora.db.adaptive_cascade_step`) share it.
"""
import math

import numpy as np

DEFAULTS = dict(start_fraction=0.5, target_recall=0.95, explore_fraction=0.05,
                batch=None, batch_fraction=0.02, hit='detector', top_fraction=0.05,
                stable_batches=2, bins=20, ucb=0.95, draws=400, prior_strength=1.0,
                order='rank', seed=0, min_batch=10)


def config(cfg=None):
    c = dict(DEFAULTS)
    c.update({k: v for k, v in (cfg or {}).items() if v is not None})
    if c['hit'] not in ('detector', 'top_fraction'):
        raise ValueError(f"adaptive.hit must be detector or top_fraction, got {c['hit']!r}")
    if c['order'] not in ('rank', 'cost'):
        raise ValueError(f"adaptive.order must be rank or cost, got {c['order']!r}")
    for k in ('target_recall', 'start_fraction', 'ucb'):
        if not 0.0 < float(c[k]) <= 1.0:
            raise ValueError(f"adaptive.{k} must be in (0, 1], got {c[k]}")
    if not 0.0 <= float(c['explore_fraction']) < 1.0:
        raise ValueError(f"adaptive.explore_fraction must be in [0, 1), got {c['explore_fraction']}")
    if int(c['stable_batches']) < 1:
        raise ValueError("adaptive.stable_batches must be >= 1")
    return c


def pava_decreasing(y, w):
    """Weighted least-squares non-increasing fit (pool adjacent violators)."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi)); wts.append(float(wi)); cnt.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


def _pava_rows(P, w):
    return np.vstack([pava_decreasing(row, w) for row in P])


class AdaptiveCascade:
    """
    State of one adaptive run over a gated population of n pairs.

    `rank`: GaussDCA rank of each pair (any increasing key; 0 = best).
    `cost`: expected GPU seconds per pair (only `order: cost` and reporting use it).
    """

    def __init__(self, rank, cost=None, cfg=None):
        self.c = config(cfg)
        self.rank = np.asarray(rank, float)
        self.n = n = len(self.rank)
        self.cost = np.ones(n) if cost is None else np.asarray(cost, float)
        self.order_rank = np.argsort(self.rank, kind='stable')
        pos = np.empty(n, int); pos[self.order_rank] = np.arange(n)
        self.pos = pos                                  # rank position of each pair
        nb = max(1, min(int(self.c['bins']), n))
        self.bin = (pos * nb) // n                      # equal-count rank bins
        self.nb = nb
        self.bin_n = np.bincount(self.bin, minlength=nb)
        self.done = np.zeros(n, bool)                   # plmDCA result known (or released)
        self.explored = np.zeros(n, bool)               # came in as an exploration draw
        self.llr = np.full(n, np.nan)
        self.det = np.zeros(n, bool)
        self.batch_size = int(self.c['batch'] or max(int(self.c['min_batch']),
                                                     round(float(self.c['batch_fraction']) * n)))
        self.rng = np.random.default_rng(int(self.c['seed']))
        self.history = []                               # one dict per estimate()
        self.seq = []                                   # frontier pairs in the order released

    # -- observations ---------------------------------------------------------

    def mark(self, idx, explored=False):
        idx = np.asarray(idx, int)
        self.done[idx] = True
        if explored:
            self.explored[idx] = True

    def observe(self, idx, llr, hit):
        idx = np.asarray(idx, int)
        self.done[idx] = True
        self.llr[idx] = np.asarray(llr, float)
        self.det[idx] = np.asarray(hit, bool)

    # -- estimator --------------------------------------------------------------

    def _weights(self, seen):
        """Horvitz-Thompson weights of the seen pairs for the population."""
        w = np.zeros(self.n)
        w[seen] = 1.0
        unseen = ~self.done
        for b in range(self.nb):
            inb = self.bin == b
            u = int((unseen & inb).sum())
            if not u:
                continue
            rep = seen & inb & self.explored
            if not rep.any():
                rep = seen & inb
            if not rep.any():                       # borrow the nearest bin with samples
                for d in range(1, self.nb):
                    for bb in (b - d, b + d):
                        if 0 <= bb < self.nb:
                            r = seen & (self.bin == bb) & self.explored
                            if r.any():
                                rep = r
                                break
                    if rep.any():
                        break
            if rep.any():
                w[rep] += u / rep.sum()
        return w

    def theta(self):
        """plmDCA log_lr_raw threshold of the top_fraction hit definition."""
        seen = self.done & ~np.isnan(self.llr)
        if not seen.any():
            return np.inf
        w = self._weights(seen)
        v, ww = self.llr[seen], w[seen]
        o = np.argsort(-v)
        cw = np.cumsum(ww[o]) / ww.sum()
        k = int(np.searchsorted(cw, float(self.c['top_fraction'])))
        return float(v[o][min(k, len(o) - 1)])

    def hits(self, th=None):
        if self.c['hit'] == 'detector':
            return self.det & self.done
        th = self.theta() if th is None else th
        return self.done & ~np.isnan(self.llr) & (self.llr >= th)

    def priority(self, pbar):
        """Processing order of the unseen pairs (best first)."""
        unseen = np.flatnonzero(~self.done)
        if self.c['order'] == 'rank':
            return unseen[np.argsort(self.pos[unseen], kind='stable')]
        key = pbar[self.bin[unseen]] / np.maximum(self.cost[unseen], 1e-9)
        return unseen[np.lexsort((self.pos[unseen], -key))]

    def estimate(self):
        """
        Re-estimate p(r) and the self-consistent threshold t*.

        The processing sequence is the frontier pairs in the order they ran, then
        the unseen pairs in the current priority order (explored pairs are not in
        it: they run whatever the threshold). For every cut f of that sequence,
        missed(f) = the hits already found among frontier pairs past f (exact: a
        tighter threshold would have lost them) + sum over unseen pairs past f of
        their bin's rate (posterior draws). t* = the smallest f with
        quantile_ucb(missed(f)) <= (1 - target_recall) x E[total hits].
        """
        c = self.c
        seen = self.done & ~np.isnan(self.llr)
        th = self.theta() if c['hit'] == 'top_fraction' else None
        h = self.hits(th) & seen
        s_b = np.bincount(self.bin[seen], minlength=self.nb).astype(float)
        h_b = np.bincount(self.bin[h], minlength=self.nb).astype(float)
        # prior mean: the Horvitz-Thompson hit rate of the population (the processed
        # pairs alone are the top ranks and would put a high prior on the tail)
        w = self._weights(seen)
        p0 = float((w * h).sum() / max(w.sum(), 1.0))
        p0 = min(max(p0, 0.5 / max(s_b.sum(), 1.0)), 0.5)
        m = float(c['prior_strength'])
        a, b = m * p0 + h_b, m * (1 - p0) + s_b - h_b
        P = self.rng.beta(a, b, size=(int(c['draws']), self.nb))
        P = _pava_rows(P, s_b + m)
        pbar = pava_decreasing(a / (a + b), s_b + m)
        order = self.priority(pbar)
        main = np.asarray(self.seq, int)
        seq = np.concatenate([main, order])
        nS, nM = len(seq), len(main)
        known = np.zeros(nS)
        known[:nM] = h[main]
        onehot = np.zeros((nS, self.nb))
        if len(order):
            onehot[np.arange(nM, nS), self.bin[order]] = 1.0
        # suffix sums: hits / unseen pairs per bin at positions >= f
        suf_known = np.concatenate([np.cumsum(known[::-1])[::-1], [0.0]])
        suf_U = np.vstack([np.cumsum(onehot[::-1], axis=0)[::-1], np.zeros((1, self.nb))])
        grid = np.unique(np.concatenate([np.linspace(0, nS, min(nS, 1000) + 1).astype(int),
                                         np.arange(max(nM - 2 * self.batch_size, 0),
                                                   min(nM + 2 * self.batch_size, nS) + 1)]))
        missed = suf_known[grid][:, None] + suf_U[grid] @ P.T      # grid x draws
        ucb = np.quantile(missed, float(c['ucb']), axis=1)
        found = float(h.sum())
        total = found + float(suf_U[nM] @ pbar)
        budget = (1.0 - float(c['target_recall'])) * total
        ok = np.flatnonzero(ucb <= budget)
        t_star = int(grid[ok[0]]) if len(ok) else nS
        i_now = int(np.searchsorted(grid, nM))
        rec = dict(n_done=int(self.done.sum()), frontier=nM, t_star=t_star, found=found,
                   total_hat=total, budget=budget, missed_ucb_now=float(ucb[i_now]),
                   recall_hat=found / total if total else 1.0, theta=th,
                   t_star_rank=float(self.rank[seq[t_star - 1]]) if t_star > 0 else 0.0,
                   frontier_rank=float(self.rank[main].max()) if nM else 0.0, p0=p0)
        self.history.append(rec)
        self._pbar, self._order = pbar, order
        return rec

    def converged(self):
        """Frontier past t*, and t* moved by at most one batch over the last k batches."""
        k = int(self.c['stable_batches'])
        if len(self.history) < k + 1:
            return False
        last = self.history[-1]
        if last['frontier'] < last['t_star']:
            return False
        ts = [r['t_star'] for r in self.history[-(k + 1):]]
        return max(ts) - min(ts) <= self.batch_size

    def exhausted(self):
        return not (~self.done).any()

    def next_batch(self):
        """Indices to run next: the frontier part (in processing order) and the
        exploration part (random unseen pairs past the frontier, stratified by bin)."""
        if not hasattr(self, '_order'):
            unseen = np.flatnonzero(~self.done)
            order = unseen[np.argsort(self.pos[unseen], kind='stable')]
        else:
            order = self._order[~self.done[self._order]]
        if not len(order):
            return np.array([], int), np.array([], int)
        nb = self.batch_size
        ne = int(math.ceil(float(self.c['explore_fraction']) * nb)) if self.c['explore_fraction'] else 0
        main = order[:max(nb - ne, 1)]
        rest = order[len(main):]
        explore = []
        if ne and len(rest):
            ex_b = np.bincount(self.bin[self.explored], minlength=self.nb).astype(float)
            un_b = np.bincount(self.bin[rest], minlength=self.nb).astype(float)
            pool = {b: list(self.rng.permutation(rest[self.bin[rest] == b])) for b in np.flatnonzero(un_b)}
            for _ in range(min(ne, len(rest))):
                live = [b for b in pool if pool[b]]
                if not live:
                    break
                b = max(live, key=lambda bb: (un_b[bb] / (ex_b[bb] + 1.0), -bb))
                explore.append(pool[b].pop()); ex_b[b] += 1; un_b[b] -= 1
        return np.asarray(main, int), np.asarray(explore, int)

    def release(self, main, explore):
        """Record a batch as handed to plmDCA (results may come later)."""
        main, explore = np.asarray(main, int), np.asarray(explore, int)
        self.seq.extend(int(i) for i in main)
        self.mark(main)
        self.mark(explore, explored=True)


def simulate(rank, llr, det, cost, cfg=None, max_batches=10_000):
    """Run the algorithm against a table with full plmDCA results; returns (state, processed mask)."""
    st = AdaptiveCascade(rank, cost, cfg)
    for _ in range(max_batches):
        main, ex = st.next_batch()
        if not len(main) and not len(ex):
            break
        st.release(main, ex)
        idx = np.concatenate([main, ex])
        st.observe(idx, llr[idx], det[idx])
        st.estimate()
        if st.converged():
            break
    return st

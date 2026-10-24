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

`model: costaware` (CostAwareThreshold, below the batch mode) is the
fast-converging alternative: stratified random convergence batches, a cost-aware
score s = P(top call | features known before plmDCA) / expected seconds, a
threshold t (share of the population with s >= s*) re-estimated after every
`check_every` completed plmDCA pairs and frozen once it changes by less than
`freeze_rel_change` on `freeze_consecutive` consecutive checks; then plmDCA runs
on every pair with s >= s*, highest s first. Below `min_population` gated pairs
the driver falls back to the batch mode (`small_population_mode: base`), and a
hit-structure guard switches to it when detector hits are much denser than on
the screen.
"""
import math

import numpy as np

DEFAULTS = dict(start_fraction=0.5, target_recall=0.95, explore_fraction=0.05,
                batch=None, batch_fraction=0.02, hit='detector', top_fraction=0.05,
                stable_batches=2, bins=20, ucb=0.95, draws=400, prior_strength=1.0,
                order='rank', seed=0, min_batch=10)


def config(cfg=None):
    model = (cfg or {}).get('model', 'batch')
    if model == 'costaware':
        return costaware_config(cfg)
    if model not in ('batch', None):
        raise ValueError(f"adaptive.model must be batch or costaware, got {model!r}")
    cfg = {k: v for k, v in (cfg or {}).items() if k != 'model'}
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
        # suffix sums: hits / unseen pairs per bin at positions >= f
        suf_known = np.concatenate([np.cumsum(known[::-1])[::-1], [0.0]])
        grid = np.unique(np.concatenate([np.linspace(0, nS, min(nS, 1000) + 1).astype(int),
                                         np.arange(max(nM - 2 * self.batch_size, 0),
                                                   min(nM + 2 * self.batch_size, nS) + 1)]))
        upos = np.arange(nM, nS)
        ubin = self.bin[order] if len(order) else np.array([], int)
        suf_U_grid = np.zeros((len(grid), self.nb))
        for b in range(self.nb):
            pb = upos[ubin == b]                        # sorted positions of bin b's unseen pairs
            suf_U_grid[:, b] = len(pb) - np.searchsorted(pb, grid, side='left')
        U_now = np.bincount(ubin, minlength=self.nb).astype(float)
        missed = suf_known[grid][:, None] + suf_U_grid @ P.T      # grid x draws
        ucb = np.quantile(missed, float(c['ucb']), axis=1)
        found = float(h.sum())
        total = found + float(U_now @ pbar)
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


# =============================================================================
# Cost model of a plmDCA pair (seconds), used by both models
# =============================================================================
#
# Headline-hours model of option A (data H100, fp16 history, maxeval 50):
# uncapped t = 7.18e-7 N^2 + 3.79e-10 B N^2, times the measured/model ratio of the
# B stratum (edges 150 / 2,000 / 5,000 / 10,000). The same constants as the
# simulator (perf/day/adaptive/sim.py cost_s) the cost-aware policy was tuned
# and tested with. N = alignment length, B = number of sequences (the GaussDCA
# source row's `length` and `number`). It ranks pairs well (Spearman 0.96 against
# measured wall time) but is about 14 % low in level; only the ordering matters.

COST_N2 = 7.18e-7
COST_BN2 = 3.79e-10
COST_B_EDGES = (150, 2000, 5000, 10000)
COST_B_RATIO = (1.190, 1.275, 1.285, 0.995, 0.801)


def cost_s(N, B):
    """Expected plmDCA seconds of a pair of length N with B sequences (vectorised)."""
    N, B = np.asarray(N, float), np.asarray(B, float)
    r = np.asarray(COST_B_RATIO)[np.searchsorted(COST_B_EDGES, B, side='right')]
    return r * (COST_N2 * N ** 2 + COST_BN2 * B * N ** 2)


# =============================================================================
# model: costaware -- the fast-converging cost-aware hand-off threshold
# =============================================================================
#
# Population: the gated GaussDCA pairs; per pair, from the source row only (all
# known before plmDCA runs): GaussDCA rank r (position by log_lr_raw among all
# ranked pairs, 0 = best), N, B, Neff/L and cost = cost_s(N, B).
#
# Convergence batches: stratified random pairs over `strata` equal-count rank
# strata; each draw goes to the stratum with the largest weight / (pairs drawn
# there + 1), weight `tilt` for strata starting above rank `tilt_above`, else 1.
# The batches never depend on the estimate.
#
# Estimate (after every `check_every` completed pairs; ~0.5 s on 917k pairs):
#   1. theta = Horvitz-Thompson (1 - top_fraction) quantile of the observed plmDCA
#      log_lr_raw (each stratum's unobserved pairs are represented by its observed
#      pairs, or by the nearest stratum's); y = 1[log_lr_raw >= theta].
#   2. p = logistic(beta . x), x = [1, log r, hinges of log r at 0.02 / 0.06 /
#      0.15 / 0.3 / 0.5, log cost, log Neff/L, log N, log B] standardised over the
#      population; HT-weighted ridge (`ridge`, not on the intercept) logistic
#      regression, Newton with line search (6 steps), warm-started from the
#      previous estimate; Cov(beta) = sandwich H^-1 J H^-1.
#   3. s = p / cost. Order the population by s. missed(f) = sum of p over the
#      unreleased pairs past the f-th; sd(f)^2 = g' Cov g + sum p(1 - p) over the
#      same pairs, g = sum p(1 - p) x (delta method); total = observed tops +
#      sum of p over pairs in flight and unreleased pairs. s* = the score of the
#      f-th pair for the smallest f with missed + z sd <= (1 - target_recall)
#      total (on a 4,001-point grid, refined inside the crossing interval);
#      t = f / n.
# Freeze: |t_k - t_(k-1)| < freeze_rel_change x t_(k-1) (or no change), t_0 =
# start_fraction, on freeze_consecutive consecutive estimates. Then beta, theta
# and s* are fixed and plmDCA runs on every unreleased pair with s >= s*,
# highest s first.
#
# Hit-structure guard: the rule targets top calls only. At every estimate before
# the freeze, once guard_min_obs pairs are observed, r = HT detector hits / HT top
# calls among the observed convergence pairs; r > guard_hits_per_top (screen:
# 0.06-0.12 at the population level, at most 0.21 along 200 simulated
# trajectories and 0.26 on 600 screen sub-populations of 7.5k-100k pairs;
# he2021: 0.70, every trajectory > 0.58) switches the run to the batch mode
# (base_k2), which continues from the convergence pairs as its exploration
# sample.
#
# This is the frozen finalist of perf/day/adaptive/fast/costaware (costaware.py,
# finalist.json: z = 1, m = 3); tests/fixtures/costaware_prototype.py is a copy
# of that prototype and tests/test_cascade_costaware.py checks parity.
# Evidence (simulation on 3,903 measured truth rows imputed to the screen's
# 917,537 gated pairs, perf/day/adaptive/fast/validate): about 49 measured
# plmDCA-h (41 h on this cost model; 56 h under iid imputation) against 108 h for
# the batch mode, conditional on the truth sample's almost top-free tail (rank >=
# 0.4: 1 top call in 838 rows); 3-6 % of the top calls in that tail cost +8 to
# +38 h with top recall >= 0.95 in only 84-90 % of runs. Per run, top-call recall
# >= 0.95 holds in about 95-98 % (other batch seeds: 175/180; screen
# sub-populations: 98/100 at every size), not with certainty. The policy's own
# recall estimate (1 - missed / total at the freeze) is uncorrelated with the
# true recall (r = 0.05) and must not be reported as a run's recall. Below
# min_population gated pairs (50,000) the driver uses the batch mode
# (small_population_mode): there the finalist meets both recalls in only
# 92-96 % of runs (base 100 %) and saves at most about 5 measured hours.

COSTAWARE_DEFAULTS = dict(model='costaware', start_fraction=0.5, check_every=50, freeze_rel_change=0.01,
                          freeze_consecutive=3, z=1.0, target_recall=0.95, top_fraction=0.05, strata=20,
                          tilt=3.0, tilt_above=0.4, ridge=1.0, min_population=50000,
                          small_population_mode='base', seed=0,
                          # hit-structure guard (guard_hits_per_top 0 = off)
                          guard_hits_per_top=0.3, guard_min_obs=500,
                          # driver: claimable convergence pairs kept released ahead of the
                          # estimate (None = check_every), pairs released per step after the
                          # freeze, seconds after which a claimed pair is reclaimed
                          lead=None, release_block=1000, running_timeout=3600.0)
COSTAWARE_KNOTS = (0.02, 0.06, 0.15, 0.3, 0.5)
COSTAWARE_NEWTON = 6
COSTAWARE_GRID = 4001
SMALL_POPULATION_MODES = ('base', 'full')
# keys that define the rule: a run refuses to continue when one of them changed
# (lead, release_block and running_timeout are operational and may change; a
# change of min_population / small_population_mode is caught by the mode check)
COSTAWARE_RULE_KEYS = ('start_fraction', 'check_every', 'freeze_rel_change', 'freeze_consecutive', 'z',
                       'target_recall', 'top_fraction', 'strata', 'tilt', 'tilt_above', 'ridge', 'seed',
                       'guard_hits_per_top', 'guard_min_obs')


def costaware_config(cfg=None):
    c = dict(COSTAWARE_DEFAULTS)
    c.update({k: v for k, v in (cfg or {}).items() if v is not None})
    unknown = set(c) - set(COSTAWARE_DEFAULTS)
    if unknown:
        raise ValueError(f"adaptive (model costaware): unknown keys {sorted(unknown)}")
    for k in ('start_fraction', 'freeze_rel_change'):
        if not 0.0 < float(c[k]) <= 1.0:
            raise ValueError(f"adaptive.{k} must be in (0, 1], got {c[k]}")
    for k in ('target_recall', 'top_fraction'):
        if not 0.0 < float(c[k]) < 1.0:
            raise ValueError(f"adaptive.{k} must be in (0, 1), got {c[k]}")
    if not 0.0 <= float(c['tilt_above']) <= 1.0:
        raise ValueError(f"adaptive.tilt_above must be in [0, 1], got {c['tilt_above']}")
    for k in ('z', 'guard_hits_per_top'):
        if not float(c[k]) >= 0.0:
            raise ValueError(f"adaptive.{k} must be >= 0, got {c[k]}")
    for k in ('tilt', 'running_timeout', 'ridge'):
        if not float(c[k]) > 0.0:
            raise ValueError(f"adaptive.{k} must be > 0, got {c[k]}")
    for k in ('check_every', 'freeze_consecutive', 'strata', 'release_block'):
        if int(c[k]) != float(c[k]) or int(c[k]) < 1:
            raise ValueError(f"adaptive.{k} must be an integer >= 1, got {c[k]}")
    if int(c['min_population']) != float(c['min_population']) or int(c['min_population']) < 0:
        raise ValueError(f"adaptive.min_population must be an integer >= 0, got {c['min_population']}")
    if c['small_population_mode'] not in SMALL_POPULATION_MODES:
        raise ValueError(f"adaptive.small_population_mode must be one of {SMALL_POPULATION_MODES}, "
                         f"got {c['small_population_mode']!r}")
    if c['lead'] is None:
        c['lead'] = int(c['check_every'])
    if int(c['lead']) != float(c['lead']) or int(c['lead']) < 1:
        raise ValueError(f"adaptive.lead must be an integer >= 1, got {c['lead']}")
    if int(c['guard_min_obs']) != float(c['guard_min_obs']) or int(c['guard_min_obs']) < 0:
        raise ValueError(f"adaptive.guard_min_obs must be an integer >= 0, got {c['guard_min_obs']}")
    for k in ('check_every', 'freeze_consecutive', 'strata', 'release_block', 'min_population', 'lead', 'seed',
              'guard_min_obs'):
        c[k] = int(c[k])
    for k in ('start_fraction', 'freeze_rel_change', 'target_recall', 'top_fraction', 'tilt_above', 'z', 'ridge',
              'tilt', 'running_timeout', 'guard_hits_per_top'):
        c[k] = float(c[k])
    return c


def costaware_design(rank, cost, neffoverL, N, B, knots=COSTAWARE_KNOTS):
    """Standardised design matrix [1, log r, hinges, log cost, log Neff/L, log N, log B]."""
    lr = np.log(np.maximum(np.asarray(rank, float), 1e-3))
    cols = [lr] + [np.maximum(0.0, lr - k) for k in np.log(knots)]
    cols.append(np.log(np.maximum(np.asarray(cost, float), 1e-12)))
    cols.append(np.log(np.maximum(np.asarray(neffoverL, float), 1e-6)))
    cols += [np.log(np.maximum(np.asarray(N, float), 1.0)), np.log(np.maximum(np.asarray(B, float), 1.0))]
    X = np.column_stack(cols)
    if np.isnan(X).any():               # a source row without Neff/L (no gate): the population mean
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-12
        Z = np.nan_to_num((X - mu) / sd, nan=0.0)
    else:
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - mu) / sd
    return np.column_stack([np.ones(len(lr)), Z])


def _sigmoid(eta):
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def _penalised_ll(X, y, w, beta, L):
    eta = X @ beta
    return float((w * (y * eta - np.logaddexp(0, eta))).sum() - 0.5 * beta @ L @ beta)


def fit_ridge_logistic(X, y, w, beta, ridge, iters=COSTAWARE_NEWTON):
    """Weighted ridge logistic regression (Newton with step halving, warm start
    `beta`). Returns beta and its sandwich covariance H^-1 J H^-1 with
    J = sum w^2 (y - p)^2 x x'."""
    L = np.eye(X.shape[1]) * float(ridge)
    L[0, 0] = 1e-6
    beta = np.asarray(beta, float).copy()
    cur = _penalised_ll(X, y, w, beta, L)
    for _ in range(int(iters)):
        p = _sigmoid(X @ beta)
        H = X.T @ (X * (w * p * (1 - p))[:, None]) + L
        g = X.T @ (w * (y - p)) - L @ beta
        step = np.linalg.solve(H, g)
        a = 1.0
        while True:
            nb = beta + a * step
            new = _penalised_ll(X, y, w, nb, L)
            if new >= cur - 1e-9 or a < 1e-3:
                break
            a /= 2
        beta, cur = nb, new
        if np.abs(a * step).max() < 1e-5:
            break
    p = _sigmoid(X @ beta)
    H = X.T @ (X * (w * p * (1 - p))[:, None]) + L
    Hinv = np.linalg.inv(H)
    J = X.T @ (X * (w ** 2 * (y - p) ** 2)[:, None])
    return beta, Hinv @ J @ Hinv


class CostAwareThreshold:
    """
    The cost-aware hand-off threshold over a gated population of n pairs.

    Stateless between calls apart from caches: `block()` draws the next
    convergence pairs, `estimate()` turns the observed results into (t, s*), so
    the DB driver rebuilds everything from its tables.
    """

    def __init__(self, rank, cost, neffoverL, N, B, cfg=None):
        self.c = c = costaware_config(cfg)
        self.rank = np.asarray(rank, float)
        self.cost = np.asarray(cost, float)
        self.n = n = len(self.rank)
        self.X = costaware_design(self.rank, self.cost, neffoverL, N, B)
        self.order_rank = np.argsort(self.rank, kind='stable')
        pos = np.empty(n, int)
        pos[self.order_rank] = np.arange(n)
        self.nb = nb = max(1, min(c['strata'], n))
        self.bin = (pos * nb) // max(n, 1)
        lo = np.array([self.rank[self.order_rank[min(int(b * n / nb), n - 1)]] for b in range(nb)]) if n else \
            np.zeros(nb)
        self.alloc = np.where(lo < c['tilt_above'], c['tilt'], 1.0)
        self.grid = np.unique(np.linspace(0, n - 1, COSTAWARE_GRID).astype(int)) if n else np.array([], int)

    def initial_beta(self):
        beta = np.zeros(self.X.shape[1])
        q = self.c['top_fraction']
        beta[0] = math.log(q / (1 - q))
        return beta

    # -- freeze rule --------------------------------------------------------------

    def check(self, t_prev, t_new):
        """(passes, relative change) of one convergence check."""
        d = abs(float(t_new) - float(t_prev))
        ok = d == 0.0 or d < self.c['freeze_rel_change'] * float(t_prev)
        return bool(ok), (d / float(t_prev) if t_prev else (0.0 if d == 0 else math.inf))

    # -- convergence batches ------------------------------------------------------

    def block(self, released, explored, rng, k):
        """k stratified random unreleased pairs (indices, in draw order).
        `released`: pairs already handed to plmDCA (in flight or finished);
        `explored`: the convergence pairs among them (the allocation counts them)."""
        left = np.flatnonzero(~np.asarray(released, bool))
        ex_b = np.bincount(self.bin[np.asarray(explored, bool)], minlength=self.nb).astype(float)
        un_b = np.bincount(self.bin[left], minlength=self.nb)
        out, taken = [], {}
        for _ in range(min(int(k), len(left))):
            live = np.flatnonzero(un_b > 0)
            if not len(live):
                break
            b = live[np.argmax(self.alloc[live] / (ex_b[live] + 1.0))]
            if b not in taken:
                taken[b] = list(rng.permutation(left[self.bin[left] == b]))
            out.append(int(taken[b].pop()))
            ex_b[b] += 1
            un_b[b] -= 1
        return np.asarray(out, int)

    # -- estimator ----------------------------------------------------------------

    def ht_weights(self, obs, avail):
        """HT weights: 1 per observed pair, plus each stratum's unobserved
        (available) pairs spread over its observed pairs (over the nearest
        stratum's if it has none)."""
        u_b = np.bincount(self.bin[avail & ~obs], minlength=self.nb).astype(float)
        e_b = np.bincount(self.bin[obs], minlength=self.nb).astype(float)
        add = np.zeros(self.nb)
        has = np.flatnonzero(e_b > 0)
        for b in np.flatnonzero(u_b > 0):
            if e_b[b] > 0:
                add[b] += u_b[b] / e_b[b]
            elif len(has):
                bb = has[np.argmin(np.abs(has - b))]
                add[bb] += u_b[b] / e_b[bb]
        w = obs.astype(float)
        w[obs] += add[self.bin[obs]]
        return w

    def scores(self, beta):
        p = _sigmoid(self.X @ np.asarray(beta, float))
        return p, p / self.cost

    def estimate(self, obs_idx, llr, released=None, failed=None, beta_prev=None):
        """
        (t, s*) from the observed pairs `obs_idx` with plmDCA log_lr_raw `llr`.
        `released`: pairs handed to plmDCA (default: the observed ones); those
        not observed and not `failed` are in flight: they will run, so they are
        never missed, and their p counts towards the total. `failed`: released
        pairs that ended without a result (left out of the population).
        `beta_prev`: the previous fit (warm start).
        """
        c, n = self.c, self.n
        obs = np.zeros(n, bool)
        obs[np.asarray(obs_idx, int)] = True
        plm = np.full(n, np.nan)
        plm[np.asarray(obs_idx, int)] = np.asarray(llr, float)
        released = obs.copy() if released is None else (np.asarray(released, bool) | obs)
        failed = np.zeros(n, bool) if failed is None else (np.asarray(failed, bool) & ~obs)
        seen = np.flatnonzero(obs)
        wall = self.ht_weights(obs, ~failed)
        v, ww = plm[seen], wall[seen]
        o = np.argsort(-v)
        cw = np.cumsum(ww[o]) / ww.sum()
        theta = float(v[o][min(int(np.searchsorted(cw, c['top_fraction'])), len(o) - 1)])
        y = (plm[seen] >= theta).astype(float)
        w = wall[seen] / wall[seen].mean()
        beta0 = self.initial_beta() if beta_prev is None else np.asarray(beta_prev, float)
        beta, cov = fit_ridge_logistic(self.X[seen], y, w, beta0, c['ridge'])
        p, score = self.scores(beta)
        un = ~released
        inflight = released & ~obs & ~failed
        found = float(y.sum())
        total = found + float(p[inflight].sum()) + float(p[un].sum())
        budget = (1.0 - c['target_recall']) * total
        o = np.argsort(-score, kind='stable')
        pu = np.where(un[o], p[o], 0.0)
        vu = pu * (1 - pu)
        m_suf = np.cumsum(pu[::-1])[::-1]
        v_suf = np.cumsum(vu[::-1])[::-1]
        G = np.cumsum((self.X[o] * vu[:, None])[::-1], axis=0)[::-1]

        def ucb(ix):
            Gi = G[ix]
            return m_suf[ix] + c['z'] * np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', Gi, cov, Gi) + v_suf[ix], 0))

        grid = self.grid
        okg = np.flatnonzero(ucb(grid) <= budget)          # a cut before f keeps positions 0..f-1
        if not len(okg):
            f = n
        elif okg[0] == 0:
            f = 0
        else:
            ix = np.arange(grid[okg[0] - 1], grid[okg[0]] + 1)
            f = int(ix[np.flatnonzero(ucb(ix) <= budget)[0]])
        s_star = float(score[o[f - 1]]) if f > 0 else math.inf
        return dict(n_obs=int(len(seen)), n_inflight=int(inflight.sum()), n_failed=int(failed.sum()),
                    t=f / n, f=int(f), s_star=s_star, theta=theta, found=found, total_hat=total,
                    missed_hat=float(m_suf[f]) if f < n else 0.0, beta=[float(b) for b in beta])

    def hit_density(self, obs_idx, llr, hit, theta, failed=None):
        """The guard statistic: HT-weighted detector hits / HT-weighted top calls
        (llr >= theta) among the observed pairs (inf if no top call was seen)."""
        n = self.n
        obs = np.zeros(n, bool)
        obs[np.asarray(obs_idx, int)] = True
        failed = np.zeros(n, bool) if failed is None else (np.asarray(failed, bool) & ~obs)
        w = self.ht_weights(obs, ~failed)
        ix = np.asarray(obs_idx, int)
        tops = float((w[ix] * (np.asarray(llr, float) >= float(theta))).sum())
        hits = float((w[ix] * np.asarray(hit, bool)).sum())
        return hits / tops if tops > 0 else (0.0 if hits == 0 else math.inf)

    def selection(self, beta, s_star):
        """Mask of the pairs the frozen rule runs, and their order (highest s first)."""
        _, score = self.scores(beta)
        sel = score >= s_star
        idx = np.flatnonzero(sel)
        return sel, idx[np.argsort(-score[idx], kind='stable')]


def simulate_costaware(rank, cost, neffoverL, N, B, llr, cfg=None, streams='single'):
    """The cost-aware policy end to end against a table with full plmDCA results,
    one convergence batch of check_every pairs at a time with no pairs in flight.
    streams 'single': one random stream seeded `seed` (the prototype's setting);
    'block': block j (1, 2, ...) draws from default_rng([seed, j]) and the
    observations are passed in release order, as the DB driver does, so a
    synchronous driver run reproduces it exactly. Returns
    (run mask, history, frozen record or None)."""
    st = CostAwareThreshold(rank, cost, neffoverL, N, B, cfg)
    c = st.c
    llr = np.asarray(llr, float)
    rng = np.random.default_rng(c['seed'])
    released = np.zeros(st.n, bool)
    obs = []
    t_prev, beta, streak, frozen, hist = c['start_fraction'], None, 0, None, []
    j = 0
    while not released.all():
        j += 1
        if streams == 'block':
            rng = np.random.default_rng([c['seed'], j])
        idx = st.block(released, released, rng, c['check_every'])
        if not len(idx):
            break
        released[idx] = True
        obs.extend(int(i) for i in idx)
        seen = np.asarray(obs) if streams == 'block' else np.flatnonzero(released)
        rec = st.estimate(seen, llr[seen], released, None, beta)
        ok, rel = st.check(t_prev, rec['t'])
        streak = streak + 1 if ok else 0
        rec.update(t_prev=t_prev, rel_change=rel, check_ok=ok, streak=streak,
                   frozen=streak >= c['freeze_consecutive'])
        hist.append(rec)
        t_prev, beta = rec['t'], rec['beta']
        if rec['frozen']:
            frozen = rec
            break
    rec = frozen or (hist[-1] if hist else None)
    run = released.copy()
    if rec is not None:
        run |= st.selection(rec['beta'], rec['s_star'])[0]
    return run, hist, frozen

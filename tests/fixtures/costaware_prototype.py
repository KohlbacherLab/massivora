# Verbatim copy of perf/day/adaptive/fast/costaware/costaware.py (the frozen cost-aware prototype),
# kept as the reference for tests/test_cascade_costaware.py::test_prototype_parity. Do not edit.
"""costaware: a fast-converging GaussDCA -> plmDCA hand-off with a cost-aware frozen selection.

Estimator (cheap, deterministic, re-run after every 50 completed plmDCA pairs; ~0.4 s on 917 k pairs):
  1. theta_hat = Horvitz-Thompson (1 - q) quantile of the observed plmDCA log_lr_raw (20 equal-count
     GaussDCA-rank strata; exploit pairs count once, each stratum's unseen pairs are represented by its
     random exploration pairs) -> y = 1[plm >= theta_hat] on the observed pairs.
  2. p(x) = logistic(beta . x), x = [1, log rank, log-rank hinges (default 0.02, 0.06, 0.15, 0.3, 0.5),
     log cost, log Neff/L, log N, log B] (all known before plmDCA runs; feats='' = rank only), ridge lam on
     the standardised slopes, HT-weighted (wfit) pseudo-likelihood, Newton with line search, warm-started
     from the previous fit.  Cov(beta) = sandwich H^-1 J H^-1 (sandwich=True) or H^-1.
  3. Score s_i = p_i / cost_i (expected top calls per GPU-second).  Order the population by s.
     missed(f) = sum of p over UNSEEN pairs below the f-th score (pairs already run count as recall);
     sd(f)^2 = g(f)' Cov(beta) g(f) + sum p(1-p), g(f) = sum p(1-p) x over the same pairs (delta method).
     total_hat = observed tops + sum of p over all unseen pairs.
     t = |{i: s_i >= s*}| / n, s* = the lowest score with missed + z sd <= (1 - R) total_hat
     ('score' mode).  The 'rank' mode does the same along GaussDCA rank (t = a rank fraction).
  Cost per estimate: one matrix-vector product, one sort and a few cumulative sums over the n pairs;
  the UCB is evaluated on a 4,001-point grid and refined inside the crossing interval.

Run (the user's rule): t_0 = 0.5; after every 50 completed pairs re-estimate t_k; freeze the first time
|t_k - t_(k-1)| / t_(k-1) < tol (default 0.01), optionally guarded (min pairs, min observed tops, or the
rule holding m consecutive times).  At the freeze, beta, theta_hat and s* are fixed and the run completes
plmDCA on every unseen pair with s_i >= s* (score mode) or rank <= t (rank mode), in descending s.
simulate() is that policy end to end.

Batches during convergence (50 pairs): a share `explore` is exploration (stratified random over the 20
rank strata, stratum weight `tilt` for strata above rank `split`, else 1), the rest is exploitation: the
unseen pairs with the highest current score inside the current threshold.  Default explore = 1 (all
random): then the batches do not depend on t at all, and the estimator sees an unbiased stratified sample.
Before the first estimate the exploit order is GaussDCA rank.

traj.py records every estimate up to `n_max` pairs; the batches never depend on the freeze rule, so
any rule can be replayed exactly on the trace (freeze.py); check_policy.py confirms simulate() == replay.
"""
import numpy as np

KN = np.log([0.02, 0.06, 0.15, 0.3, 0.5])
DEF = dict(q=0.05, R=0.95, z=1.645, lam=1.0, batch=50, explore=1.0, tilt=3.0, split=0.4, nbins=20,
           feats='cnNB', t0=0.5, wfit=True, sandwich=True, n_max=20000, seed=0, newton=6, knots=None, zs=(), seed_off=0)


def design(rank, cost, nl, N, B, feats='cnNB', knots=None):
    kn = KN if knots is None else np.log(knots)
    z = np.log(np.maximum(rank, 1e-3))
    cols = [z] + [np.maximum(0, z - k) for k in kn]
    if 'c' in feats:
        cols.append(np.log(cost))
    if 'n' in feats:
        cols.append(np.log(nl))
    if 'NB' in feats:
        cols += [np.log(N), np.log(B)]
    X = np.column_stack(cols)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    return np.column_stack([np.ones(len(z)), (X - mu) / sd])


def _ll(X, y, w, beta, L):
    eta = X @ beta
    return float((w * (y * eta - np.logaddexp(0, eta))).sum() - 0.5 * beta @ L @ beta)


def fit_logit(X, y, w, beta, lam, it=6, sandwich=True):
    """Weighted ridge logistic regression (Newton + line search). Returns beta and Cov(beta):
    the sandwich H^-1 J H^-1 (J = sum w^2 (y-p)^2 x x') or H^-1."""
    L = np.eye(X.shape[1]) * lam; L[0, 0] = 1e-6
    cur = _ll(X, y, w, beta, L)
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        H = X.T @ (X * (w * p * (1 - p))[:, None]) + L
        g = X.T @ (w * (y - p)) - L @ beta
        step = np.linalg.solve(H, g)
        a = 1.0
        while True:
            nb = beta + a * step
            new = _ll(X, y, w, nb, L)
            if new >= cur - 1e-9 or a < 1e-3:
                break
            a /= 2
        beta, cur = nb, new
        if np.abs(a * step).max() < 1e-5:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
    H = X.T @ (X * (w * p * (1 - p))[:, None]) + L
    Hinv = np.linalg.inv(H)
    if not sandwich:
        return beta, Hinv
    J = X.T @ (X * (w ** 2 * (y - p) ** 2)[:, None])
    return beta, Hinv @ J @ Hinv


class CostAware:
    def __init__(self, rank, cost, nl, N, B, cfg=None):
        self.c = c = dict(DEF, **(cfg or {}))
        self.rank, self.cost = np.asarray(rank, float), np.asarray(cost, float)
        self.n = n = len(self.rank)
        self.X = design(self.rank, self.cost, nl, N, B, c['feats'], c['knots'])
        self.order_rank = np.argsort(self.rank, kind='stable')
        pos = np.empty(n, int); pos[self.order_rank] = np.arange(n)
        self.pos = pos
        self.nb = nb = int(min(c['nbins'], n))
        self.bin = (pos * nb) // n
        self.bin_n = np.bincount(self.bin, minlength=nb)
        lo = np.array([self.rank[self.order_rank[min(int(b * n / nb), n - 1)]] for b in range(nb)])
        self.alloc = np.where(lo < float(c['split']), float(c['tilt']), 1.0)
        self.done = np.zeros(n, bool); self.explored = np.zeros(n, bool)
        self.plm = np.full(n, np.nan)
        self.rng = np.random.default_rng(int(c['seed']))
        self.beta = np.zeros(self.X.shape[1]); self.beta[0] = np.log(c['q'] / (1 - c['q']))
        self.t = float(c['t0'])
        self.sel_score = None          # current selection cut on the score (score mode)
        self.score = None              # current score of every pair (None = rank order)
        self.p = None
        self.history = []

    # -- batches ------------------------------------------------------------------------------------
    def next_batch(self, mode='score'):
        c = self.c
        B = int(c['batch'])
        ne = int(round(float(c['explore']) * B))
        nx = B - ne
        unseen = np.flatnonzero(~self.done)
        if not len(unseen):
            return np.array([], int), np.array([], int)
        # exploit: best current score inside the current threshold
        if self.score is None or mode == 'rank':
            cand = unseen[self.rank[unseen] <= self.t] if self.score is None or mode == 'rank' else unseen
            key = self.pos[cand]
        else:
            cand = unseen[self.score[unseen] >= self.sel_score]
            key = -self.score[cand]
        if len(cand) > nx:
            cand = cand[np.argpartition(key, nx)[:nx]]
        exploit = cand
        # explore: stratified random over rank bins (tilted allocation)
        left = np.setdiff1d(unseen, exploit, assume_unique=True)
        ex_b = np.bincount(self.bin[self.explored], minlength=self.nb).astype(float)
        un_b = np.bincount(self.bin[left], minlength=self.nb)
        explore = []
        taken = {}
        for _ in range(min(ne + max(nx - len(exploit), 0), len(left))):
            live = np.flatnonzero(un_b > 0)
            if not len(live):
                break
            b = live[np.argmax(self.alloc[live] / (ex_b[live] + 1.0))]
            if b not in taken:
                pool = left[self.bin[left] == b]
                taken[b] = list(self.rng.permutation(pool))
            explore.append(taken[b].pop()); ex_b[b] += 1; un_b[b] -= 1
        return np.asarray(exploit, int), np.asarray(explore, int)

    def observe(self, exploit, explore, plm):
        self.done[exploit] = True; self.done[explore] = True; self.explored[explore] = True
        idx = np.concatenate([exploit, explore])
        self.plm[idx] = plm[idx]

    # -- estimator ----------------------------------------------------------------------------------
    def ht_weights(self):
        """HT weights of the seen pairs: 1 each, plus each stratum's unseen count spread over its
        exploration pairs (nearest stratum with exploration pairs if it has none)."""
        seen = self.done
        u_b = np.bincount(self.bin[~seen], minlength=self.nb).astype(float)
        e_b = np.bincount(self.bin[seen & self.explored], minlength=self.nb).astype(float)
        add = np.zeros(self.nb)                 # extra weight per exploration pair, by its own stratum
        has = np.flatnonzero(e_b > 0)
        for b in np.flatnonzero(u_b > 0):
            if e_b[b] > 0:
                add[b] += u_b[b] / e_b[b]
            elif len(has):
                bb = has[np.argmin(np.abs(has - b))]
                add[bb] += u_b[b] / e_b[bb]
        w = seen.astype(float)
        ex = seen & self.explored
        w[ex] += add[self.bin[ex]]
        return w

    def theta(self, w):
        seen = self.done
        v, ww = self.plm[seen], w[seen]
        o = np.argsort(-v)
        cw = np.cumsum(ww[o]) / ww.sum()
        k = int(np.searchsorted(cw, float(self.c['q'])))
        return float(v[o][min(k, len(o) - 1)])

    def estimate(self):
        c = self.c
        seen = np.flatnonzero(self.done)
        wall = self.ht_weights()
        th = self.theta(wall)
        y = (self.plm[seen] >= th).astype(float)
        w = wall[seen] / wall[seen].mean() if c['wfit'] else np.ones(len(seen))
        self.beta, Hinv = fit_logit(self.X[seen], y, w, self.beta.copy(), float(c['lam']), int(c['newton']),
                                    bool(c['sandwich']))
        p = 1 / (1 + np.exp(-np.clip(self.X @ self.beta, -30, 30)))
        self.p = p
        un = ~self.done
        found = float(y.sum())
        total = found + float(p[un].sum())
        budget = (1 - float(c['R'])) * total
        zs = [float(c['z'])] + [float(x) for x in c['zs'] if float(x) != float(c['z'])]
        score = p / self.cost
        out = dict(n_seen=len(seen), theta=th, found=found, total_hat=total)
        grid = np.unique(np.linspace(0, self.n - 1, 4001).astype(int))
        for mode in ('score', 'rank'):
            o = np.argsort(-score, kind='stable') if mode == 'score' else self.order_rank
            u = un[o]
            pu = np.where(u, p[o], 0.0)
            vu = pu * (1 - pu)
            # suffix sums from position f to the end (missed if the cut sits before f)
            m_suf = np.cumsum(pu[::-1])[::-1]
            v_suf = np.cumsum(vu[::-1])[::-1]
            G = np.cumsum((self.X[o] * vu[:, None])[::-1], axis=0)[::-1]

            def sd_at(ix):
                Gi = G[ix]
                return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', Gi, Hinv, Gi) + v_suf[ix], 0))
            sdg = sd_at(grid)
            for z in zs:
                okg = np.flatnonzero(m_suf[grid] + z * sdg <= budget)   # cut before f keeps 0..f-1
                if not len(okg):
                    f = self.n
                elif okg[0] == 0:
                    f = 0
                else:
                    ix = np.arange(grid[okg[0] - 1], grid[okg[0]] + 1)
                    ok = np.flatnonzero(m_suf[ix] + z * sd_at(ix) <= budget)
                    f = int(ix[ok[0]])
                sfx = '' if z == zs[0] else f'@{z:g}'
                if mode == 'score':
                    out['t_score' + sfx] = f / self.n
                    out['s_star' + sfx] = float(score[o[f - 1]]) if f > 0 else np.inf
                    if not sfx:
                        out['missed_hat_score'] = float(m_suf[f]) if f < self.n else 0.0
                else:
                    out['t_rank' + sfx] = float(self.rank[o[f - 1]]) if f > 0 else 0.0
            if mode == 'score':
                self._o_score = o
        self.zs = zs
        self.score = score
        self.history.append(out)
        return out

    def set_threshold(self, mode):
        last = self.history[-1]
        if mode == 'score':
            self.t = last['t_score']; self.sel_score = last['s_star']
        else:
            self.t = last['t_rank']

    def selection(self, rec, mode, sfx=''):
        """Pairs the frozen rule runs (seen or not) for an estimate record + its score vector."""
        if mode == 'score':
            return self.score >= rec['s_star' + sfx]
        return self.rank <= rec['t_rank' + sfx]


def simulate(rank, cost, nl, N, B, plm, cfg=None, mode='score', tol=0.01, kind='rel', m=1, n_min=0, h_min=0,
             top=None):
    """The policy end to end: converge (t_0 = 0.5, re-estimate per batch, freeze rule), then complete
    plmDCA on every pair inside the frozen threshold.  `top` (optional, truth) only for reporting.
    Returns (state, run mask, info)."""
    st = CostAware(rank, cost, nl, N, B, cfg)
    prev, streak, frozen = float(st.c['t0']), 0, None
    while not st.done.all():
        ex, xp = st.next_batch(mode)
        if not len(ex) and not len(xp):
            break
        st.observe(ex, xp, plm)
        rec = st.estimate()
        st.set_threshold(mode)
        t = rec['t_score'] if mode == 'score' else rec['t_rank']
        d = abs(t - prev)
        ok = (d < tol * prev or d == 0) if kind == 'rel' else d < tol
        streak = streak + 1 if ok else 0
        prev = t
        obs_tops = rec['found']
        if streak >= m and st.done.sum() >= n_min and obs_tops >= h_min:
            frozen = rec
            break
    rec = frozen or st.history[-1]
    conv_mask = st.done.copy()
    run = st.selection(rec, mode) | conv_mask
    info = dict(converged=frozen is not None, k=len(st.history), n_conv=int(conv_mask.sum()),
                h_conv=float(cost[conv_mask].sum() / 3600), h=float(cost[run].sum() / 3600),
                t=rec['t_score'] if mode == 'score' else rec['t_rank'], s_star=rec.get('s_star'),
                theta=rec['theta'], beta=st.beta.tolist())
    return st, run, info

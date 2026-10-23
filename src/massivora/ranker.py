"""
Cascade ranker: how likely a GaussDCA-scored pair is to be a plmDCA call.

`couple.cascade.rank_by: cascade_score` ranks a GaussDCA run by this score
instead of `log_lr_raw`. It is a gradient-boosted tree model on the GaussDCA
inter-protein block (calibrator evidence, score quantiles, top-10 mean,
focality) and the alignment depth (N, B, Beff, Neff/L, chain lengths), fitted
on the he2021 benchmark against plmDCA's top-5 % calls and detector hits.
On the H. pylori screen, its top 16 % keeps as many plmDCA calls as the top
30 % by `log_lr_raw`, at about half the plmDCA cost.

The model ships as JSON trees (scikit-learn's HistGradientBoostingClassifier
nodes) and is evaluated here with numpy, so scikit-learn is not needed.
"""
import json
import os
from functools import lru_cache

import numpy as np

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'cascade_ranker_v1.json')


@lru_cache(maxsize=1)
def _model():
    with open(MODEL_FILE) as fh:
        m = json.load(fh)
    m['trees'] = [{k: np.asarray(v) for k, v in t.items()} for t in m['trees']]
    return m


def predict(X):
    """Decision function (log-odds) of the model for feature rows X (n, 14); NaN means missing."""
    m = _model()
    # Rounded as in training: discrete features (g_conc10) carry 1-ulp noise
    X = np.round(np.atleast_2d(np.asarray(X, dtype=np.float64)), 9)
    rows = np.arange(len(X))
    out = np.full(len(X), m['baseline'])
    for t in m['trees']:
        node = np.zeros(len(X), dtype=np.int64)
        while True:
            inner = ~t['is_leaf'][node].astype(bool)
            if not inner.any():
                break
            x = X[rows, t['feature_idx'][node]]
            left = np.where(np.isnan(x), t['missing_go_to_left'][node].astype(bool),
                            x <= t['num_threshold'][node])
            node = np.where(inner, np.where(left, t['left'][node], t['right'][node]), node)
        out += t['value'][node]
    return out


def _focality(M):
    s = np.linalg.svd(M - M.mean(), compute_uv=False)
    return float(s[0] / np.sqrt((s ** 2).sum()))


def features(analyzer, calibrator, N_eff, B, log_lr_raw):
    """The model's 14 features for one pair, in `_model()['features']` order."""
    G = analyzer.NormalizedCouplings(N_eff).to_numpy(dtype=float)
    f = G[np.isfinite(G)]
    gs = np.sort(f)[::-1]
    N = int(analyzer.raw_couplings.shape[0])
    LA, LB = G.shape
    return [log_lr_raw, (f >= 0).mean(), (f >= 0.001).mean(), gs[0],
            calibrator._feature(G, f, 'conc_entropy_10'), gs[:10].mean(),
            _focality(np.nan_to_num(G)), (f >= 1).mean(),
            np.log(N), np.log(B), np.log(N_eff), np.log(N_eff / N + 1e-3), np.log(LA), np.log(LB)]


def cascade_score(analyzer, calibrator, N_eff, B, log_lr_raw):
    """Ranker log-odds for one GaussDCA-scored pair (higher = send to plmDCA first)."""
    return float(predict(features(analyzer, calibrator, N_eff, B, log_lr_raw))[0])

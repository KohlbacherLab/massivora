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


def _stored_features(row, output_dir, prefix):
    """Features of one stored GaussDCA pair: its score from the shards, its
    column layout from the pair's zarr attributes (the matrix is not read)."""
    import zarr
    from massivora.analyzer import BuiltinGaussCalibrator, ComplexAnalyzer
    from massivora.db import get_hashed_file_path
    from massivora.utils import read_score_record
    pair_id, name, file_hash, B, Beff, log_lr_raw, file_id, offset, length = row
    try:
        S = read_score_record(output_dir, prefix, file_id, offset, length)['matrix']
        filename, _ = get_hashed_file_path(name, file_hash, output_dir)
        attrs = zarr.open_group(filename, mode='r')['align'].attrs
        analyzer = ComplexAnalyzer(np.asarray(S, np.float64), saved_columns=attrs['saved_columns'],
                                   chainA_length=attrs['lengthes'][0], EC_threshold=0)
        return pair_id, features(analyzer, BuiltinGaussCalibrator(), Beff, B, log_lr_raw)
    except Exception as e:
        return pair_id, repr(e)[:200]


def backfill(db_path, output_dir, table='couplings', prefix='couplings', processes=None,
             chunk=20000):
    """
    Add cascade_score to the DONE pairs of a GaussDCA run scored before the
    ranker existed, from the stored scores and alignments. Resumable: only
    rows without a score are read. Returns (scored, failed).
    """
    import logging
    import sqlite3
    from multiprocessing import Pool

    from massivora.db import quote_identifier
    t = quote_identifier(table)
    conn = sqlite3.connect(db_path, timeout=600)
    try:
        conn.execute(f'ALTER TABLE {t} ADD COLUMN cascade_score REAL')
    except sqlite3.OperationalError:
        pass                                        # already there
    rows = conn.execute(
        f'SELECT id, name, file_hash, number, effnumber, log_lr_raw, score_file_id, score_offset, '
        f'score_len FROM {t} WHERE status = 0 AND cascade_score IS NULL '
        f'AND score_file_id IS NOT NULL ORDER BY id').fetchall()
    scored = failed = 0
    with Pool(processes) as pool:
        for s in range(0, len(rows), chunk):
            got = pool.starmap(_stored_features, [(r, output_dir, prefix) for r in rows[s:s + chunk]],
                               chunksize=16)
            ok = [(i, f) for i, f in got if not isinstance(f, str)]
            for i, err in ((i, f) for i, f in got if isinstance(f, str)):
                logging.warning(f"cascade_score: pair {i} not scored: {err}")
            if ok:
                scores = predict(np.array([f for _, f in ok], dtype=np.float64))
                conn.executemany(f'UPDATE {t} SET cascade_score = ? WHERE id = ?',
                                 [(float(v), i) for v, (i, _) in zip(scores, ok)])
                conn.commit()
            scored += len(ok)
            failed += len(got) - len(ok)
            logging.info(f"cascade_score: {scored + failed} of {len(rows)} pairs, {failed} failed")
    conn.close()
    return scored, failed


if __name__ == '__main__':
    import argparse
    import logging
    ap = argparse.ArgumentParser(description='Backfill cascade_score into a GaussDCA run (SQLite).')
    ap.add_argument('db', help='SQLite file of the GaussDCA run')
    ap.add_argument('output_dir', help="the run's couplings directory (score shards and pair zarrs)")
    ap.add_argument('--table', default='couplings')
    ap.add_argument('--processes', type=int, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    print('scored %d, failed %d' % backfill(a.db, a.output_dir, a.table, processes=a.processes))

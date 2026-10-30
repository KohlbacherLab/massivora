"""couple.cascade.adaptive: the self-consistent GaussDCA -> plmDCA threshold."""
import sqlite3

import numpy as np
import pytest

from massivora.adaptive_cascade import AdaptiveCascade, pava_decreasing, simulate
from massivora.db import STATUS, adaptive_cascade_step, apply_cascade, create_job_db


def _truth(n, seed=0):
    """Hits concentrated at the top of the GaussDCA ranking, a thin tail below."""
    rng = np.random.default_rng(seed)
    r = (np.arange(n) + 0.5) / n
    p = np.where(r < 0.1, 0.5, np.where(r < 0.25, 0.08, 0.002))
    hit = rng.random(n) < p
    llr = np.where(hit, 5 + rng.random(n), rng.normal(0, 1, n))
    return r, llr, hit


def test_pava_is_non_increasing_and_keeps_mean():
    y, w = np.array([0.1, 0.3, 0.2, 0.05, 0.07]), np.array([1, 2, 1, 1, 3])
    f = pava_decreasing(y, w)
    assert np.all(np.diff(f) <= 1e-12)
    assert np.isclose((f * w).sum(), (y * w).sum())


@pytest.mark.parametrize('hit', ['detector', 'top_fraction'])
def test_simulation_converges_below_full_and_keeps_hits(hit):
    r, llr, det = _truth(4000)
    st = simulate(r, llr, det, np.ones(len(r)), dict(hit=hit, top_fraction=0.05, seed=1))
    assert st.converged()
    assert st.done.mean() < 0.9                       # stopped before running everything
    truth = det if hit == 'detector' else llr >= np.quantile(llr, 0.95)
    assert (truth & st.done).sum() / truth.sum() >= 0.9
    last = st.history[-1]
    assert last['frontier'] >= last['t_star']
    assert st.explored.sum() > 0 and st.explored.sum() < st.done.sum()


def test_exploration_reaches_below_the_frontier():
    r, llr, det = _truth(2000)
    st = AdaptiveCascade(r, None, dict(seed=0, batch=100, explore_fraction=0.2))
    main, ex = st.next_batch()
    assert len(main) == 80 and len(ex) == 20
    assert st.pos[ex].min() > st.pos[main].max()
    assert len(set(st.bin[ex])) >= 10                 # stratified over the rank bins


def test_config_validation():
    with pytest.raises(ValueError):
        AdaptiveCascade([0.1, 0.2], None, dict(hit='nope'))
    with pytest.raises(ValueError):
        AdaptiveCascade([0.1, 0.2], None, dict(target_recall=1.5))


# -- DB driver -------------------------------------------------------------------

def _cfg(tmp_path, **adaptive):
    return {
        'project': {'project_path': str(tmp_path)},
        'database': {'backend': 'sqlite', 'filename': 'plm.sqlite3', 'shared_filesystem': False},
        'align': {},
        'couple': {'cascade': {'source_db': 'gauss.sqlite3', 'min_neffoverL': 0.1,
                               'adaptive': {'batch': 40, 'seed': 3, **adaptive}}},
    }


@pytest.fixture
def project(tmp_path):
    for name in ('gauss', 'plm'):
        create_job_db({**_cfg(tmp_path), 'database': {
            'backend': 'sqlite', 'filename': f'{name}.sqlite3', 'shared_filesystem': False}})
    n = 1200
    r, llr, hit = _truth(n, seed=5)
    g = sqlite3.connect(tmp_path / 'gauss.sqlite3')
    # every 10th pair is below the depth gate
    g.executemany("INSERT INTO couplings (name, pid1, pid2, status, log_lr_raw, neffoverL, length, number) "
                  "VALUES (?, 'A', 'B', ?, ?, ?, 300, 1000)",
                  [(f'p{i:04d}', STATUS['DONE'], float(n - i), 0.05 if i % 10 == 7 else 1.0) for i in range(n)])
    g.commit()
    p = sqlite3.connect(tmp_path / 'plm.sqlite3')
    p.executemany("INSERT INTO couplings (name, pid1, pid2, status) VALUES (?, 'A', 'B', ?)",
                  [(f'p{i:04d}', STATUS['NOOPT']) for i in range(n)])
    p.commit()
    return tmp_path, {f'p{i:04d}': (llr[i], bool(hit[i])) for i in range(n)}


def _status(path):
    return dict(sqlite3.connect(path).execute("SELECT name, status FROM couplings"))


def _work(path, truth):
    """Stand-in for plmDCA: finish every claimable pair with its known result."""
    c = sqlite3.connect(path)
    rows = c.execute("SELECT name FROM couplings WHERE status = ?", (STATUS['NOOPT'],)).fetchall()
    c.executemany("UPDATE couplings SET status = ?, log_lr_raw = ?, high_precision_hit = ? WHERE name = ?",
                  [(STATUS['DONE'], truth[n][0], int(truth[n][1]), n) for (n,) in rows])
    c.commit()
    return len(rows)


def test_first_batch_and_gate(project):
    path, truth = project
    cfg = _cfg(path)
    ranked, kept, changed = apply_cascade(cfg)
    st = _status(path / 'plm.sqlite3')
    noopt = sorted(n for n, s in st.items() if s == STATUS['NOOPT'])
    assert ranked == 1200 and kept == len(noopt) == 40
    assert all(int(n[1:]) % 10 != 7 for n in noopt)                    # gated pairs never released
    assert all(st[f'p{i:04d}'] == STATUS['SKIPPED'] for i in range(7, 1200, 10))
    # idempotent: a second worker start releases nothing new
    assert apply_cascade(cfg)[1] == 40
    # a step while released pairs are still claimable waits
    assert adaptive_cascade_step(cfg)['status'] == 'wait'


def test_runs_to_a_stable_threshold(project):
    path, truth = project
    cfg = _cfg(path, hit='detector', explore_fraction=0.1)
    apply_cascade(cfg)
    for _ in range(200):
        _work(path / 'plm.sqlite3', truth)
        step = adaptive_cascade_step(cfg)
        if step['status'] == 'stopped':
            break
    assert step['status'] == 'stopped'
    st = _status(path / 'plm.sqlite3')
    gated = [n for n in truth if int(n[1:]) % 10 != 7]
    run = {n for n in gated if st[n] == STATUS['DONE']}
    hits = {n for n in gated if truth[n][1]}
    assert len(run) < len(gated)
    assert len(hits & run) / len(hits) >= 0.9
    log = sqlite3.connect(path / 'plm.sqlite3').execute(
        "SELECT frontier, t_star, stop FROM cascade_adaptive_log ORDER BY batch").fetchall()
    assert log[-1][2] == 1 and log[-1][0] >= log[-1][1]
    # once stopped, further steps and worker starts change nothing
    assert adaptive_cascade_step(cfg)['status'] == 'stopped'
    before = _status(path / 'plm.sqlite3')
    apply_cascade(cfg)
    assert _status(path / 'plm.sqlite3') == before


def test_rejects_keep_rule_with_adaptive(project):
    path, _ = project
    cfg = _cfg(path)
    cfg['couple']['cascade']['keep_fraction'] = 0.2
    with pytest.raises(ValueError):
        apply_cascade(cfg)

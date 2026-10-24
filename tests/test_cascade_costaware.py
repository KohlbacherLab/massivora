"""couple.cascade.adaptive, model: costaware -- the fast-converging cost-aware hand-off threshold."""
import importlib.util
import json
import os
import sqlite3
import time

import numpy as np
import pytest

import massivora.db as mdb
from massivora.adaptive_cascade import (CostAwareThreshold, config, cost_s, costaware_config,
                                        simulate_costaware)
from massivora.db import STATUS, adaptive_cascade_step, adaptive_mode, apply_cascade, create_job_db

FINALIST = dict(model='costaware', z=1.0, freeze_consecutive=3, freeze_rel_change=0.01, check_every=50,
                start_fraction=0.5, target_recall=0.95, top_fraction=0.05, strata=20, tilt=3.0, tilt_above=0.4,
                ridge=1.0)


def _population(n, seed=0):
    """Ranks, depth features and a plmDCA score that rises with GaussDCA rank and depth."""
    rng = np.random.default_rng(seed)
    rank = (np.arange(n) + 0.5) / n
    N = rng.integers(80, 1500, n).astype(float)
    B = np.exp(rng.uniform(np.log(40), np.log(40000), n)).round()
    nl = np.maximum(np.exp(rng.normal(0.0, 1.0, n)), 0.12)
    llr = -1.6 * np.log(rank) + 0.8 * np.log(nl) + rng.normal(0, 1.4, n)
    return rank, N, B, nl, llr


def _prototype():
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'costaware_prototype.py')
    spec = importlib.util.spec_from_file_location('costaware_prototype', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- estimator ---------------------------------------------------------------------

def test_cost_model_constants():
    # the simulator's cost_s (perf/day/adaptive/sim.py): ratio by B stratum x (a N^2 + b B N^2)
    N, B = np.array([500., 500., 500., 500., 500.]), np.array([100., 1000., 3000., 8000., 20000.])
    base = 7.18e-7 * N ** 2 + 3.79e-10 * B * N ** 2
    assert np.allclose(cost_s(N, B), base * np.array([1.190, 1.275, 1.285, 0.995, 0.801]))
    assert np.isclose(cost_s(500, 150), 1.275 * (7.18e-7 * 500 ** 2 + 3.79e-10 * 150 * 500 ** 2))


@pytest.mark.parametrize('n,seed', [(6000, 0), (20000, 1), (12000, 7)])
def test_prototype_parity(n, seed):
    """Massivora's estimator and the frozen prototype on the same population and
    observations: the same batches, t trajectory, s* and freeze."""
    proto = _prototype()
    rank, N, B, nl, llr = _population(n, seed)
    cost = cost_s(N, B)
    cfg = dict(FINALIST, seed=seed)
    run, hist, frozen = simulate_costaware(rank, cost, nl, N, B, llr, cfg)
    pcfg = dict(q=0.05, R=0.95, z=1.0, zs=(), lam=1.0, batch=50, explore=1.0, tilt=3.0, split=0.4, nbins=20,
                feats='cnNB', knots=None, t0=0.5, wfit=True, sandwich=True, newton=6, seed=seed)
    st, prun, info = proto.simulate(rank, cost, nl, N, B, llr, pcfg, mode='score', tol=0.01, kind='rel', m=3)
    assert len(hist) == len(st.history)
    t_m = np.array([h['t'] for h in hist])
    t_p = np.array([h['t_score'] for h in st.history])
    assert np.allclose(t_m, t_p, rtol=0, atol=1e-12)
    assert np.allclose([h['s_star'] for h in hist], [h['s_star'] for h in st.history], rtol=1e-9)
    assert np.allclose([h['theta'] for h in hist], [h['theta'] for h in st.history])
    assert (frozen is not None) == info['converged']
    assert np.array_equal(run, prun)
    assert np.allclose(frozen['beta'], info['beta'], rtol=1e-9, atol=1e-12)
    # the freeze rule: 3 consecutive relative changes < 1 % (t_0 = 0.5), first time
    ok = [h['check_ok'] for h in hist]
    assert ok[-3:] == [True, True, True]
    assert not any(all(ok[i:i + 3]) for i in range(len(ok) - 3))


def test_estimate_is_independent_of_observation_order():
    rank, N, B, nl, llr = _population(5000, 3)
    st = CostAwareThreshold(rank, cost_s(N, B), nl, N, B, FINALIST)
    rng = np.random.default_rng(0)
    obs = rng.choice(5000, 600, replace=False)
    a = st.estimate(obs, llr[obs])
    p = rng.permutation(600)
    b = st.estimate(obs[p], llr[obs][p])
    assert a['t'] == b['t'] and np.isclose(a['s_star'], b['s_star'])


def test_pairs_in_flight_are_not_missed():
    """A pair released but without a result yet runs anyway: it is never counted
    as missed, so t can only go down (s* up) when pairs are in flight."""
    rank, N, B, nl, llr = _population(5000, 4)
    st = CostAwareThreshold(rank, cost_s(N, B), nl, N, B, FINALIST)
    rng = np.random.default_rng(1)
    obs = rng.choice(5000, 800, replace=False)
    a = st.estimate(obs, llr[obs])
    rel = np.zeros(5000, bool)
    rel[obs] = True
    _, score = st.scores(a['beta'])
    top = np.argsort(-score)[:300]
    rel[top] = True                                   # the best-scored pairs are in flight
    b = st.estimate(obs, llr[obs], released=rel)
    assert b['n_inflight'] == int((rel.sum() - 800))
    assert b['t'] <= a['t']


def test_estimate_takes_about_a_second_on_a_screen():
    n = 917_537
    rank, N, B, nl, llr = _population(n, 5)
    t0 = time.perf_counter()
    st = CostAwareThreshold(rank, cost_s(N, B), nl, N, B, FINALIST)
    t_init = time.perf_counter() - t0
    rng = np.random.default_rng(2)
    obs = rng.choice(n, 2000, replace=False)
    st.estimate(obs, llr[obs])
    times = []
    beta = None
    for _ in range(3):
        t0 = time.perf_counter()
        beta = st.estimate(obs, llr[obs], beta_prev=beta)['beta']
        times.append(time.perf_counter() - t0)
    print(f"init {t_init:.2f} s, estimate {min(times):.2f}-{max(times):.2f} s on {n} pairs")
    assert min(times) < 1.5


def test_missing_depth_takes_the_population_mean():
    rank, N, B, nl, llr = _population(3000, 6)
    nl = nl.copy()
    nl[::7] = np.nan                                  # source rows without Neff/L (no gate set)
    st = CostAwareThreshold(rank, cost_s(N, B), nl, N, B, FINALIST)
    assert np.isfinite(st.X).all() and np.allclose(st.X[::7, 8], 0.0)
    obs = np.arange(0, 3000, 5)
    assert np.isfinite(st.estimate(obs, llr[obs])['t'])


def test_config_validation():
    assert costaware_config(dict(model='costaware'))['lead'] == 50
    assert config(dict(model='costaware'))['min_population'] == 10000
    assert config(dict(model='costaware'))['small_population_mode'] == 'base'
    for bad in (dict(keep_fraction=0.3), dict(batch=40), dict(explore_fraction=0.1), dict(start_fraction=0),
                dict(target_recall=1.0), dict(freeze_consecutive=0), dict(check_every=2.5), dict(z=-1),
                dict(small_population_mode='B_x05'), dict(min_population=-1), dict(tilt_above=1.5),
                dict(strata=0), dict(freeze_rel_change=0)):
        with pytest.raises(ValueError):
            costaware_config(dict(model='costaware', **bad))
    with pytest.raises(ValueError):
        config(dict(model='fast'))
    # the batch mode keeps its keys and defaults
    c = config(dict(batch=40, explore_fraction=0.1))
    assert c['batch'] == 40 and c['stable_batches'] == 2 and 'model' not in c
    assert config(dict(model='batch', batch=30))['batch'] == 30


# -- DB driver -----------------------------------------------------------------------

N_ALL = 22000           # ranked source rows; every 11th is below the depth gate -> 20,000 gated pairs


def _cfg(tmp_path, **adaptive):
    a = dict(FINALIST, min_population=0, seed=3)
    a.update(adaptive)
    return {
        'project': {'project_path': str(tmp_path)},
        'database': {'backend': 'sqlite', 'filename': 'plm.sqlite3', 'shared_filesystem': False},
        'align': {},
        'couple': {'cascade': {'source_db': 'gauss.sqlite3', 'min_neffoverL': 0.1, 'adaptive': a}},
    }


def _gated(i):
    return i % 11 != 5


def _make_project(tmp_path, seed=5):
    for name in ('gauss', 'plm'):
        create_job_db({**_cfg(tmp_path), 'database': {
            'backend': 'sqlite', 'filename': f'{name}.sqlite3', 'shared_filesystem': False}})
    rank, N, B, nl, llr = _population(N_ALL, seed)
    g = sqlite3.connect(tmp_path / 'gauss.sqlite3')
    g.executemany("INSERT INTO couplings (name, pid1, pid2, status, log_lr_raw, neffoverL, length, number) "
                  "VALUES (?, 'A', 'B', ?, ?, ?, ?, ?)",
                  [(f'p{i:05d}', STATUS['DONE'], float(N_ALL - i), float(nl[i]) if _gated(i) else 0.05,
                    int(N[i]), int(B[i])) for i in range(N_ALL)])
    g.commit()
    g.close()
    p = sqlite3.connect(tmp_path / 'plm.sqlite3')
    p.executemany("INSERT INTO couplings (name, pid1, pid2, status) VALUES (?, 'A', 'B', ?)",
                  [(f'p{i:05d}', STATUS['NOOPT']) for i in range(N_ALL)])
    p.commit()
    p.close()
    return {f'p{i:05d}': float(llr[i]) for i in range(N_ALL)}


@pytest.fixture
def project(tmp_path):
    return tmp_path, _make_project(tmp_path)


def _clear_caches():
    mdb._POPULATION_CACHE.clear()
    mdb._COSTAWARE_CACHE.clear()


def _db(path):
    return sqlite3.connect(path / 'plm.sqlite3')


def _work(path, truth, limit=None, now=None):
    """Stand-in for plmDCA: claim and finish claimable pairs with their known result."""
    c = _db(path)
    q = "SELECT name FROM couplings WHERE status IN (?, ?) AND claimed_at IS NULL ORDER BY id"
    rows = [r[0] for r in c.execute(q, (STATUS['NOOPT'], STATUS['PENDING'])).fetchall()]
    rows = rows if limit is None else rows[:limit]
    c.executemany("UPDATE couplings SET status = ?, log_lr_raw = ?, claimed_at = NULL WHERE name = ?",
                  [(STATUS['DONE'], truth[n], n) for n in rows])
    c.commit()
    c.close()
    return rows


def _log(path):
    c = _db(path)
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute("SELECT * FROM cascade_adaptive_log ORDER BY k")]
    c.close()
    return rows


def _released(path):
    c = _db(path)
    rows = c.execute("SELECT seq, batch, name, role FROM cascade_adaptive ORDER BY seq").fetchall()
    c.close()
    return rows


def _run_to_freeze(path, truth, cfg, max_steps=400, between=None):
    for _ in range(max_steps):
        _work(path, truth)
        if between:
            between()
        step = adaptive_cascade_step(cfg)
        if step['frozen']:
            return step
    raise AssertionError('no freeze')


def _model(path, cfg):
    pop = mdb._adaptive_population(cfg, cfg['couple']['cascade'])
    c = costaware_config(cfg['couple']['cascade']['adaptive'])
    return pop, CostAwareThreshold(pop['rank'], pop['cost'], pop['neffoverL'], pop['N'], pop['B'], c)


def test_starts_at_half_with_a_stratified_block(project):
    path, truth = project
    cfg = _cfg(path)
    assert adaptive_mode(cfg) == 'costaware'
    ranked, kept, _ = apply_cascade(cfg)
    assert ranked == N_ALL and kept == 50
    log = _log(path)
    assert len(log) == 1 and log[0]['k'] == 0 and log[0]['t'] == 0.5 and log[0]['frozen'] == 0
    rel = _released(path)
    assert len(rel) == 50 and {r[3] for r in rel} == {'explore'}
    st = dict(_db(path).execute("SELECT name, status FROM couplings"))
    assert all(_gated(int(n[1:])) for _, _, n, _ in rel)
    assert sum(s == STATUS['NOOPT'] for s in st.values()) == 50
    assert all(st[f'p{i:05d}'] == STATUS['SKIPPED'] for i in range(N_ALL) if not _gated(i))
    # stratified over the 20 rank strata, three times the weight above rank 0.4
    pop, model = _model(path, cfg)
    idx = np.array([pop['names'].index(n) for _, _, n, _ in rel])
    per = np.bincount(model.bin[idx], minlength=20)
    assert (per > 0).sum() == 20
    tilted = model.alloc == 3.0
    assert per[tilted].mean() > 2 * per[~tilted].mean()
    # idempotent: a second worker start and a step with pairs still claimable release nothing
    assert apply_cascade(cfg)[1] == 50
    assert adaptive_cascade_step(cfg)['status'] == 'wait'


def test_estimates_every_check_every_completed_pairs(project):
    path, truth = project
    cfg = _cfg(path, lead=150)
    apply_cascade(cfg)
    assert len(_released(path)) == 150
    _work(path, truth, limit=120)
    step = adaptive_cascade_step(cfg)
    assert step['estimates'] == 2                       # 120 results: two estimates, 20 wait
    c = _db(path)
    ks = dict(c.execute("SELECT k, COUNT(*) FROM cascade_adaptive_obs GROUP BY k"))
    assert ks == {1: 50, 2: 50}
    log = _log(path)
    assert [r['k'] for r in log] == [0, 1, 2]
    assert [r['n_obs'] for r in log[1:]] == [50, 100]
    assert log[1]['n_inflight'] == 100 and log[2]['n_inflight'] == 50
    assert log[1]['t_prev'] == 0.5 and log[2]['t_prev'] == log[1]['t']
    # the 20 left over and 30 more: one more estimate
    _work(path, truth, limit=30)
    assert adaptive_cascade_step(cfg)['estimates'] == 1
    assert _log(path)[-1]['n_obs'] == 150
    assert adaptive_cascade_step(cfg)['estimates'] == 0


def test_freezes_after_three_small_changes_then_stays_fixed(project):
    path, truth = project
    cfg = _cfg(path)
    apply_cascade(cfg)
    step = _run_to_freeze(path, truth, cfg)
    log = _log(path)
    est = log[1:]
    fr = [r for r in est if r['frozen']]
    assert len(fr) == 1 and fr[0] is est[-1]
    # the rule: |t_k - t_(k-1)| < 1 % of t_(k-1) on 3 consecutive estimates, first time
    t = [0.5] + [r['t'] for r in est]
    ok = [abs(t[i + 1] - t[i]) < 0.01 * t[i] or t[i + 1] == t[i] for i in range(len(est))]
    assert [bool(r['check_ok']) for r in est] == ok
    assert ok[-3:] == [True, True, True]
    assert not any(all(ok[i:i + 3]) for i in range(len(ok) - 3))
    assert [r['streak'] for r in est][-3:] == [1, 2, 3]
    assert all(r['n_obs'] == 50 * r['k'] for r in est)
    # synchronous worker: no pairs in flight at each estimate -> the estimator replays the log exactly
    pop, model = _model(path, cfg)
    c = _db(path)
    obs = c.execute("SELECT o.k, o.name, t.log_lr_raw FROM cascade_adaptive_obs o JOIN couplings t "
                    "ON t.name = o.name WHERE o.k > 0 ORDER BY o.k").fetchall()
    ix = {n: i for i, n in enumerate(pop['names'])}
    beta = None
    for r in est:
        sel = [(ix[n], l) for k, n, l in obs if k <= r['k']]
        e = model.estimate([i for i, _ in sel], [l for _, l in sel], beta_prev=beta)
        assert e['t'] == r['t'] and np.isclose(e['s_star'], r['s_star'])
        beta = e['beta']
    frozen = fr[0]
    assert json.loads(frozen['beta']) == beta
    # fixed afterwards: finishing everything adds no estimate and t, s*, beta stay
    for _ in range(50):
        _work(path, truth)
        s = adaptive_cascade_step(cfg)
        assert s['frozen'] and s['estimates'] == 0 and s['t'] == frozen['t']
        if s['status'] == 'done':
            break
    assert s['status'] == 'done'
    log2 = _log(path)
    assert len(log2) == len(log) and log2[-1]['t'] == frozen['t'] and log2[-1]['s_star'] == frozen['s_star']
    # what ran: the convergence pairs plus every pair with s >= s*
    sel, _ = model.selection(json.loads(frozen['beta']), frozen['s_star'])
    st = dict(c.execute("SELECT name, status FROM couplings"))
    done = {n for n, s_ in st.items() if s_ == STATUS['DONE']}
    rel = _released(path)
    conv = {n for _, _, n, r in rel if r == 'explore'}
    want = {pop['names'][i] for i in np.flatnonzero(sel)} | conv
    assert done == want
    assert frozen['n_selected'] == int(sel.sum())
    assert len(done) < len(pop['names'])


def test_release_after_freeze_is_highest_score_first(project):
    path, truth = project
    cfg = _cfg(path, release_block=97)
    apply_cascade(cfg)
    _run_to_freeze(path, truth, cfg)
    frozen = [r for r in _log(path) if r['frozen']][0]
    pop, model = _model(path, cfg)
    _, score = model.scores(json.loads(frozen['beta']))
    blocks = []
    for _ in range(500):
        rel = [r for r in _released(path) if r[3] == 'final']
        if not blocks or len(rel) > sum(map(len, blocks)):
            blocks.append([n for _, _, n, _ in rel][sum(map(len, blocks)):])
        _work(path, truth)
        if adaptive_cascade_step(cfg)['status'] == 'done':
            break
    final = [n for _, _, n, r in _released(path) if r == 'final']
    ix = {n: i for i, n in enumerate(pop['names'])}
    s = score[[ix[n] for n in final]]
    assert len(final) > 97 and np.all(np.diff(s) <= 0)          # released in descending s
    assert all(len(b) <= 97 for b in blocks)
    assert s.min() >= frozen['s_star']


def test_withdraws_unclaimed_convergence_pairs_below_s_star_at_freeze(project):
    path, truth = project
    cfg = _cfg(path, lead=400)
    apply_cascade(cfg)
    for _ in range(400):
        _work(path, truth, limit=60)             # 60 of >= 400 claimable: many stay unclaimed
        step = adaptive_cascade_step(cfg)
        if step['frozen']:
            break
    assert step['frozen']
    fr = [r for r in _log(path) if r['frozen']][0]
    assert fr['withdrawn'] > 0
    pop, model = _model(path, cfg)
    sel, _ = model.selection(json.loads(fr['beta']), fr['s_star'])
    st = dict(_db(path).execute("SELECT name, status FROM couplings"))
    for _, _, n, role in _released(path):
        if role == 'explore' and st[n] == STATUS['NOOPT']:
            assert sel[pop['names'].index(n)]


def test_restart_reproduces_the_run(tmp_path):
    """A fresh worker for every step (caches cleared, extra no-op steps) runs the same
    trajectory, freezes at the same estimate and releases the same pairs."""
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir(), b.mkdir()
    truth = _make_project(a)
    _make_project(b)
    cfg_a, cfg_b = _cfg(a), _cfg(b)
    apply_cascade(cfg_a)
    _clear_caches()
    apply_cascade(cfg_b)

    def restart():
        _clear_caches()
        apply_cascade(cfg_b)                       # a worker start
        adaptive_cascade_step(cfg_b)               # a step with nothing new

    _run_to_freeze(a, truth, cfg_a)
    _clear_caches()
    _run_to_freeze(b, truth, cfg_b, between=restart)
    la, lb = _log(a), _log(b)
    keys = ('k', 'n_obs', 't', 's_star', 'theta', 'beta', 'streak', 'frozen', 'withdrawn', 'n_selected')
    assert [[r[k] for k in keys] for r in la] == [[r[k] for k in keys] for r in lb]
    # after the freeze: restart between every step; the frozen state is read back, never recomputed
    for _ in range(100):
        _clear_caches()
        _work(a, truth)
        sa = adaptive_cascade_step(cfg_a)
        restart()
        _work(b, truth)
        sb = adaptive_cascade_step(cfg_b)
        if sa['status'] == 'done' and sb['status'] == 'done':
            break
    assert [r[2:] for r in _released(a)] == [r[2:] for r in _released(b)]
    assert _log(b)[-1]['s_star'] == la[-1]['s_star'] and len(_log(b)) == len(la)


def test_stuck_running_pairs_count_as_in_flight_then_are_reclaimed(project):
    path, truth = project
    cfg = _cfg(path, lead=100, running_timeout=600)
    apply_cascade(cfg)
    c = _db(path)
    names = [r[0] for r in c.execute("SELECT name FROM couplings WHERE status = ? ORDER BY id LIMIT 10",
                                     (STATUS['NOOPT'],))]
    # 10 pairs claimed by a worker that died: RUNNING, claimed 5 minutes ago
    c.executemany("UPDATE couplings SET status = ?, claimed_at = ?, job_id = 'dead' WHERE name = ?",
                  [(STATUS['RUNNING'], time.time() - 300, n) for n in names])
    c.commit()
    _work(path, truth)
    step = adaptive_cascade_step(cfg)
    assert step['estimates'] == 1                    # they do not block the estimate
    log = _log(path)
    assert log[-1]['n_inflight'] >= 10 and log[-1]['reclaimed'] == 0
    st = dict(c.execute("SELECT name, status FROM couplings"))
    assert all(st[n] == STATUS['RUNNING'] for n in names)
    # past running_timeout they are handed back (no attempt spent) and run again
    c.executemany("UPDATE couplings SET claimed_at = ? WHERE name = ?", [(time.time() - 700, n) for n in names])
    c.commit()
    adaptive_cascade_step(cfg)
    st = dict(c.execute("SELECT name, status FROM couplings"))
    assert all(st[n] == STATUS['NOOPT'] for n in names)
    assert c.execute("SELECT COUNT(*) FROM couplings WHERE claimed_at IS NOT NULL").fetchone()[0] == 0
    got = _work(path, truth)
    assert set(names) <= set(got)


def test_failed_pairs_are_left_out_not_waited_for(project):
    path, truth = project
    cfg = _cfg(path)
    apply_cascade(cfg)
    c = _db(path)
    names = [r[0] for r in c.execute("SELECT name FROM couplings WHERE status = ? ORDER BY id LIMIT 5",
                                     (STATUS['NOOPT'],))]
    c.executemany("UPDATE couplings SET status = ? WHERE name = ?", [(STATUS['FAILED'], n) for n in names])
    c.commit()
    _work(path, truth)
    adaptive_cascade_step(cfg)
    _work(path, truth)
    step = adaptive_cascade_step(cfg)
    assert step['estimates'] >= 1
    assert dict(c.execute("SELECT name, k FROM cascade_adaptive_obs WHERE k = -1")) == {n: -1 for n in names}
    assert _log(path)[-1]['n_failed'] == 5


def test_small_population_falls_back_to_the_batch_mode(project):
    path, truth = project
    cfg = _cfg(path, min_population=50000)            # 20,000 gated pairs < 50,000
    assert adaptive_mode(cfg) == 'base'
    apply_cascade(cfg)
    c = _db(path)
    cols = [d[1] for d in c.execute("PRAGMA table_info(cascade_adaptive_log)")]
    assert 'stop' in cols and 'frozen' not in cols
    rel = _released(path)
    # the base_k2 loop: batch = 2 % of the population (400), 5 % of it exploration
    assert len(rel) == 400 and sum(r[3] == 'explore' for r in rel) == 20
    for _ in range(300):
        _work(path, truth)
        step = adaptive_cascade_step(cfg)
        assert step['mode'] == 'base'
        if step['status'] == 'stopped':
            break
    assert step['status'] == 'stopped'
    st = dict(c.execute("SELECT name, status FROM couplings"))
    gated = [n for n in truth if _gated(int(n[1:]))]
    run = {n for n in gated if st[n] == STATUS['DONE']}
    q = np.quantile([truth[n] for n in gated], 0.95)
    top = {n for n in gated if truth[n] >= q}
    assert len(run) < len(gated) and len(top & run) / len(top) >= 0.9
    # a DB that ran one mode refuses the other
    cfg2 = _cfg(path, min_population=0)
    with pytest.raises(ValueError):
        adaptive_cascade_step(cfg2)


def test_small_population_full_mode_runs_every_gated_pair(project):
    path, truth = project
    cfg = _cfg(path, min_population=50000, small_population_mode='full', release_block=50000)
    assert adaptive_mode(cfg) == 'full'
    ranked, kept, _ = apply_cascade(cfg)
    assert kept == 20000
    log = _log(path)
    assert len(log) == 1 and log[0]['frozen'] == 1 and log[0]['mode'] == 'full'
    _work(path, truth)
    assert adaptive_cascade_step(cfg)['status'] == 'done'


def test_rejects_keep_rule_and_unknown_keys(project):
    path, _ = project
    cfg = _cfg(path)
    cfg['couple']['cascade']['keep_fraction'] = 0.2
    with pytest.raises(ValueError):
        apply_cascade(cfg)
    cfg = _cfg(path, explore_fraction=0.05)
    with pytest.raises(ValueError):
        apply_cascade(cfg)

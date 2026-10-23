"""GaussDCA -> plmDCA cascade: `couple.cascade` keeps the top of a GaussDCA run claimable."""
import sqlite3

import pytest

from massivora.db import STATUS, apply_cascade, create_job_db
from massivora.massiveilance import _project_is_complete


def _cfg(tmp_path, **cascade):
    return {
        'project': {'project_path': str(tmp_path)},
        'database': {'backend': 'sqlite', 'filename': 'plm.sqlite3', 'shared_filesystem': False},
        'align': {}, 'couple': {'cascade': {'source_db': 'gauss.sqlite3', **cascade}},
    }


def _fill(path, rows):
    conn = sqlite3.connect(path)
    conn.executemany("INSERT INTO couplings (name, pid1, pid2, status, log_lr_raw, high_precision_hit) "
                     "VALUES (?, 'A', 'B', ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def _status(path):
    return dict(sqlite3.connect(path).execute("SELECT name, status FROM couplings"))


@pytest.fixture
def project(tmp_path):
    for name in ('gauss', 'plm'):
        create_job_db({**_cfg(tmp_path), 'database': {
            'backend': 'sqlite', 'filename': f'{name}.sqlite3', 'shared_filesystem': False}})
    D, F, N = STATUS['DONE'], STATUS['FAILED'], STATUS['NOOPT']
    # p0..p9 ranked by log_lr_raw descending; p9 is a detector hit; px failed in GaussDCA
    _fill(tmp_path / 'gauss.sqlite3',
          [(f'p{i}', D, 10.0 - i, int(i == 9)) for i in range(10)] + [('px', F, None, 0)])
    _fill(tmp_path / 'plm.sqlite3',
          [(f'p{i}', N, None, 0) for i in range(10)] + [('px', N, None, 0)])
    # p1 already ran under plmDCA and must stay DONE
    sqlite3.connect(tmp_path / 'plm.sqlite3').execute(
        "UPDATE couplings SET status = ? WHERE name = 'p1'", (D,)).connection.commit()
    return tmp_path


def test_keeps_top_fraction_and_unranked_pairs(project):
    assert apply_cascade(_cfg(project, keep_fraction=0.3)) == (10, 3, 7)
    st = _status(project / 'plm.sqlite3')
    assert [n for n, s in sorted(st.items()) if s == STATUS['NOOPT']] == ['p0', 'p2', 'px']
    assert st['p1'] == STATUS['DONE']
    assert all(st[f'p{i}'] == STATUS['SKIPPED'] for i in range(3, 10))


def test_is_idempotent_and_widening_hands_pairs_back(project):
    apply_cascade(_cfg(project, keep_fraction=0.3))
    assert apply_cascade(_cfg(project, keep_fraction=0.3))[2] == 0
    apply_cascade(_cfg(project, keep_fraction=0.5))
    st = _status(project / 'plm.sqlite3')
    assert st['p3'] == st['p4'] == STATUS['NOOPT'] and st['p5'] == STATUS['SKIPPED']


def test_detector_ranking_puts_hits_first(project):
    apply_cascade(_cfg(project, keep_fraction=0.2, rank_by='high_precision_hit'))
    st = _status(project / 'plm.sqlite3')
    assert st['p9'] == st['p0'] == STATUS['NOOPT'] and st['p2'] == STATUS['SKIPPED']


def test_skipped_pairs_count_as_finished(project):
    cfg = _cfg(project, keep_fraction=0.3)
    apply_cascade(cfg)
    assert not _project_is_complete(cfg, 'couple')
    conn = sqlite3.connect(project / 'plm.sqlite3')
    conn.execute("UPDATE couplings SET status = ? WHERE status = ?", (STATUS['DONE'], STATUS['NOOPT']))
    conn.commit()
    assert _project_is_complete(cfg, 'couple')


@pytest.mark.parametrize('bad', [{'keep_fraction': 0}, {'keep_fraction': 1.5},
                                 {'rank_by': 'pcontact; DROP TABLE couplings'},
                                 {'source_db': 'plm.sqlite3'}])
def test_rejects_bad_settings(project, bad):
    with pytest.raises(ValueError):
        apply_cascade(_cfg(project, **bad))


def test_no_cascade_is_a_no_op(project):
    cfg = _cfg(project)
    del cfg['couple']['cascade']
    assert apply_cascade(cfg) is None


def test_executor_claims_only_the_kept_pairs(project):
    from massivora.executors.couple import BaseCouplingExecutor
    cfg = {**_cfg(project, keep_fraction=0.3),
           'paths': {'monomers': 'msa', 'couplings': 'ecs'},
           'concatenate': {'downsample': {}}}
    executor = BaseCouplingExecutor(cfg)
    try:
        claimed = {row['name'] for row in executor._claim_batch('job', 100)}
    finally:
        executor.close_db_connection()
    assert claimed == {'p0', 'p2', 'px'}


# -- depth gate: couple.cascade.min_neffoverL -------------------------------

def _set_depth(path, depths):
    conn = sqlite3.connect(path)
    conn.executemany("UPDATE couplings SET neffoverL = ? WHERE name = ?",
                     [(d, n) for n, d in depths.items()])
    conn.commit()
    conn.close()


@pytest.fixture
def gated_project(project):
    # kept by rank at 0.5: p0..p4; p1 and p3 are shallow, p7 (not kept) is shallow too,
    # p2 has no depth recorded
    _set_depth(project / 'gauss.sqlite3',
               {'p0': 2.0, 'p1': 0.05, 'p2': None, 'p3': 0.01, 'p4': 0.5,
                'p5': 1.0, 'p6': 3.0, 'p7': 0.02, 'p8': 0.9, 'p9': 0.3})
    return project


def test_gate_skips_only_shallow_kept_pairs_and_unranked_rows(gated_project):
    # 7 rows change: p3 (depth), p5..p9 (rank), px (unranked); p1 is DONE and stays
    assert apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1)) == (10, 3, 7)
    st = _status(gated_project / 'plm.sqlite3')
    # p0, p2 (NULL depth: ungated), p4 stay claimable; p1 already ran
    assert [n for n, s in sorted(st.items()) if s == STATUS['NOOPT']] == ['p0', 'p2', 'p4']
    assert st['p1'] == STATUS['DONE']
    assert st['p3'] == STATUS['SKIPPED']                       # kept by rank, gated by depth
    assert all(st[f'p{i}'] == STATUS['SKIPPED'] for i in range(5, 10))
    assert st['px'] == STATUS['SKIPPED']                       # unranked (FAILED in GaussDCA)


def test_gate_null_is_todays_behaviour(gated_project):
    assert apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=None)) == (10, 5, 5)
    st = _status(gated_project / 'plm.sqlite3')
    assert [n for n, s in sorted(st.items()) if s == STATUS['NOOPT']] == ['p0', 'p2', 'p3', 'p4', 'px']


def test_gate_is_idempotent_and_lowering_it_hands_pairs_back(gated_project):
    apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1))
    assert apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1))[2] == 0
    apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.005))
    st = _status(gated_project / 'plm.sqlite3')
    assert st['p3'] == STATUS['NOOPT'] and st['p7'] == STATUS['SKIPPED'] and st['px'] == STATUS['SKIPPED']
    # removing the gate hands the unranked row back as well
    apply_cascade(_cfg(gated_project, keep_fraction=0.5))
    assert _status(gated_project / 'plm.sqlite3')['px'] == STATUS['NOOPT']


def test_gate_zero_keeps_everything_the_cut_keeps(gated_project):
    assert apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.0))[1] == 5


def test_source_without_depth_column_is_not_gated(gated_project):
    # a GaussDCA run from before the neffoverL column existed
    conn = sqlite3.connect(gated_project / 'gauss.sqlite3')
    conn.execute("CREATE TABLE old_run (id INTEGER PRIMARY KEY, name TEXT, status INTEGER, "
                 "log_lr_raw REAL, high_precision_hit BOOLEAN)")
    conn.execute("INSERT INTO old_run SELECT id, name, status, log_lr_raw, high_precision_hit FROM couplings")
    conn.commit()
    conn.close()
    assert apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1,
                              source_table='old_run')) == (10, 5, 6)
    st = _status(gated_project / 'plm.sqlite3')
    assert st['p3'] == STATUS['NOOPT'] and st['px'] == STATUS['SKIPPED']


def test_rejects_negative_gate(gated_project):
    with pytest.raises(ValueError):
        apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=-1))


def test_executor_does_not_claim_gated_pairs(gated_project):
    from massivora.executors.couple import BaseCouplingExecutor
    cfg = {**_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1),
           'paths': {'monomers': 'msa', 'couplings': 'ecs'},
           'concatenate': {'downsample': {}}}
    executor = BaseCouplingExecutor(cfg)
    try:
        claimed = {row['name'] for row in executor._claim_batch('job', 100)}
    finally:
        executor.close_db_connection()
    assert claimed == {'p0', 'p2', 'p4'}

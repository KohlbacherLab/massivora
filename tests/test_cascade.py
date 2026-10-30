"""GaussDCA -> plmDCA cascade: `couple.cascade` keeps the top of a GaussDCA run claimable."""
import logging
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
                                 {'source_db': 'plm.sqlite3', 'keep_fraction': 0.3},
                                 # a block that neither cuts, gates nor skips does nothing
                                 {}])
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


def test_gate_skips_only_shallow_kept_pairs(gated_project):
    # 6 rows change: p3 (depth), p5..p9 (rank); p1 is DONE and stays, px is unranked
    assert apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1)) == (10, 3, 6)
    st = _status(gated_project / 'plm.sqlite3')
    # p0, p2 (NULL depth: ungated), p4 stay claimable; p1 already ran
    assert [n for n, s in sorted(st.items()) if s == STATUS['NOOPT']] == ['p0', 'p2', 'p4', 'px']
    assert st['p1'] == STATUS['DONE']
    assert st['p3'] == STATUS['SKIPPED']                       # kept by rank, gated by depth
    assert all(st[f'p{i}'] == STATUS['SKIPPED'] for i in range(5, 10))
    # the gate is a DEPTH gate: a pair the source never finished is unranked, not
    # shallow, and stays claimable unless skip_unranked says otherwise
    assert st['px'] == STATUS['NOOPT']


def test_gate_null_is_todays_behaviour(gated_project):
    assert apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=None)) == (10, 5, 5)
    st = _status(gated_project / 'plm.sqlite3')
    assert [n for n, s in sorted(st.items()) if s == STATUS['NOOPT']] == ['p0', 'p2', 'p3', 'p4', 'px']


def test_gate_is_idempotent_and_lowering_it_hands_pairs_back(gated_project):
    apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1))
    assert apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1))[2] == 0
    apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.005))
    st = _status(gated_project / 'plm.sqlite3')
    assert st['p3'] == STATUS['NOOPT'] and st['p7'] == STATUS['SKIPPED']
    apply_cascade(_cfg(gated_project, keep_fraction=0.5))
    assert _status(gated_project / 'plm.sqlite3')['p3'] == STATUS['NOOPT']


def test_gate_zero_is_a_no_op(gated_project):
    """0 must disable the gate, not merely keep every pair the cut keeps: it is
    the natural way to switch the gate off in a config that cannot write null."""
    assert apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.0))[1] == 5
    ungated = _status(gated_project / 'plm.sqlite3')
    apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=None))
    assert _status(gated_project / 'plm.sqlite3') == ungated


# -- failing open is loud ---------------------------------------------------

def test_source_without_depth_column_is_not_gated_but_warns(gated_project, caplog):
    """A gate that gates nothing is a 2-3x cost blowout, not a no-op."""
    conn = sqlite3.connect(gated_project / 'gauss.sqlite3')
    conn.execute("CREATE TABLE old_run (id INTEGER PRIMARY KEY, name TEXT, status INTEGER, "
                 "log_lr_raw REAL, high_precision_hit BOOLEAN)")
    conn.execute("INSERT INTO old_run SELECT id, name, status, log_lr_raw, high_precision_hit FROM couplings")
    conn.commit()
    conn.close()
    with caplog.at_level(logging.WARNING):
        assert apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1,
                                  source_table='old_run')) == (10, 5, 5)
    assert 'no neffoverL column' in caplog.text and 'NO pair is gated' in caplog.text
    assert _status(gated_project / 'plm.sqlite3')['p3'] == STATUS['NOOPT']


def test_null_depth_rows_are_not_gated_but_warn(gated_project, caplog):
    with caplog.at_level(logging.WARNING):
        apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1))
    # p2 is kept by rank and has NULL neffoverL
    assert '1 of 5 kept pairs have NULL neffoverL' in caplog.text
    assert _status(gated_project / 'plm.sqlite3')['p2'] == STATUS['NOOPT']


# -- skip_unranked: a coverage decision, not a depth gate -------------------

def _add_running_source_row(path, name='pr'):
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO couplings (name, pid1, pid2, status) VALUES (?, 'A', 'B', ?)",
                 (name, STATUS['RUNNING']))
    conn.commit()
    conn.close()


def test_unranked_rows_stay_claimable_by_default(gated_project):
    apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1))
    assert _status(gated_project / 'plm.sqlite3')['px'] == STATUS['NOOPT']


def test_skip_unranked_skips_failed_source_rows(gated_project):
    apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1, skip_unranked=True))
    assert _status(gated_project / 'plm.sqlite3')['px'] == STATUS['SKIPPED']
    # and it hands them back when switched off again
    apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1))
    assert _status(gated_project / 'plm.sqlite3')['px'] == STATUS['NOOPT']


def test_skip_unranked_never_skips_running_source_rows(gated_project):
    """Every worker calls apply_cascade at start, so skipping a pair the source
    is still working on would make the kept set depend on timing."""
    _add_running_source_row(gated_project / 'gauss.sqlite3')
    _fill(gated_project / 'plm.sqlite3', [('pr', STATUS['NOOPT'], None, 0)])
    apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1, skip_unranked=True))
    st = _status(gated_project / 'plm.sqlite3')
    assert st['pr'] == STATUS['NOOPT'] and st['px'] == STATUS['SKIPPED']


def test_skip_unranked_works_without_a_gate(gated_project):
    apply_cascade(_cfg(gated_project, keep_fraction=0.5, skip_unranked=True))
    st = _status(gated_project / 'plm.sqlite3')
    assert st['px'] == STATUS['SKIPPED'] and st['p3'] == STATUS['NOOPT']


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
    assert claimed == {'p0', 'p2', 'p4', 'px'}


# -- keep_count / keep_denominator, gate-only blocks, SKIPPED source rows -----
# The GaussDCA pre-gate (couple.cascade.min_neffoverL read by the GaussDCA
# stage) leaves pairs below the gate SKIPPED with no score, so the plmDCA
# cascade sees a smaller ranked population and must still select the same set.

def _claimable(path):
    return {n for n, s in _status(path).items() if s == STATUS['NOOPT']}


def test_keep_count_keeps_that_many_ranked_pairs(project):
    assert apply_cascade(_cfg(project, keep_count=3)) == (10, 3, 7)
    assert _claimable(project / 'plm.sqlite3') == {'p0', 'p2', 'px'}
    # idempotent, and a larger count hands pairs back
    assert apply_cascade(_cfg(project, keep_count=3))[2] == 0
    apply_cascade(_cfg(project, keep_count=5))
    st = _status(project / 'plm.sqlite3')
    assert st['p3'] == st['p4'] == STATUS['NOOPT'] and st['p5'] == STATUS['SKIPPED']


def test_keep_count_above_the_ranked_population_keeps_every_ranked_pair(project):
    assert apply_cascade(_cfg(project, keep_count=1000)) == (10, 10, 0)


def test_keep_count_and_keep_fraction_are_two_spellings_of_one_cut(project):
    apply_cascade(_cfg(project, keep_fraction=0.3))
    by_fraction = _status(project / 'plm.sqlite3')
    assert apply_cascade(_cfg(project, keep_count=3))[2] == 0
    assert _status(project / 'plm.sqlite3') == by_fraction


@pytest.mark.parametrize('bad', [
    {'keep_count': 0}, {'keep_count': -1}, {'keep_count': 2.5}, {'keep_count': True},
    {'keep_count': '3'},
    {'keep_count': 3, 'keep_fraction': 0.3},                 # exactly one of the two
    {'keep_count': 3, 'keep_denominator': 'ranked'},         # the denominator is keep_fraction's
    {'keep_fraction': 0.3, 'keep_denominator': 'all'},
])
def test_rejects_bad_keep_rules(project, bad):
    with pytest.raises(ValueError):
        apply_cascade(_cfg(project, **bad))


def test_gate_only_block_keeps_everything_above_the_gate(gated_project):
    # no keep rule: all 10 ranked pairs are kept, then p1, p3, p7 are gated;
    # p1 already ran under plmDCA, so 2 rows change; p2 (NULL depth) is ungated
    assert apply_cascade(_cfg(gated_project, min_neffoverL=0.1)) == (10, 7, 2)
    assert _claimable(gated_project / 'plm.sqlite3') == {'p0', 'p2', 'p4', 'p5', 'p6', 'p8', 'p9', 'px'}


def test_gate_only_block_without_a_source_run_is_a_no_op(gated_project):
    """The GaussDCA stage's own config carries only min_neffoverL (the pre-gate,
    applied while scoring); apply_cascade has nothing to rank and must not raise."""
    cfg = _cfg(gated_project, min_neffoverL=0.1)
    del cfg['couple']['cascade']['source_db']
    before = _status(gated_project / 'plm.sqlite3')
    assert apply_cascade(cfg) is None
    assert _status(gated_project / 'plm.sqlite3') == before
    # a keep rule against the stage's own table is still a misconfiguration
    cfg['couple']['cascade']['keep_count'] = 3
    with pytest.raises(ValueError):
        apply_cascade(cfg)


def test_gauss_stage_executor_accepts_a_gate_only_block(gated_project):
    from massivora.executors.couple import BaseCouplingExecutor
    cfg = {**_cfg(gated_project, min_neffoverL=0.1),
           'paths': {'monomers': 'msa', 'couplings': 'ecs'},
           'concatenate': {'downsample': {}}}
    del cfg['couple']['cascade']['source_db']
    executor = BaseCouplingExecutor(cfg)
    try:
        claimed = {row['name'] for row in executor._claim_batch('job', 100)}
    finally:
        executor.close_db_connection()
    assert claimed == {f'p{i}' for i in range(10) if i != 1} | {'px'}


def _pregate_source(path, gate=0.1):
    """What the GaussDCA pre-gate leaves behind: pairs below the gate are SKIPPED
    with their depth and no score (executors.couple._skipped_row)."""
    conn = sqlite3.connect(path)
    conn.execute("UPDATE couplings SET status = ?, log_lr_raw = NULL, high_precision_hit = 0 "
                 "WHERE status = ? AND neffoverL < ?", (STATUS['SKIPPED'], STATUS['DONE'], gate))
    conn.commit()
    conn.close()


@pytest.fixture
def pregated_project(gated_project):
    # p1, p3, p7 are SKIPPED in the source; 7 pairs stay ranked; px is FAILED
    _pregate_source(gated_project / 'gauss.sqlite3')
    return gated_project


@pytest.mark.parametrize('cascade', [
    {'keep_fraction': 0.5, 'min_neffoverL': 0.1},
    {'keep_fraction': 0.5},
    {'keep_fraction': 0.5, 'skip_unranked': False},
    {'keep_count': 4},
    {'min_neffoverL': 0.1},
    {'skip_unranked': True},
])
def test_skipped_source_rows_are_never_claimable(pregated_project, cascade):
    """A pair the source declined is a decision, not an unfinished result: it
    stays SKIPPED here whatever the gate and skip_unranked say, while an
    unfinished (FAILED) source row follows skip_unranked as before."""
    apply_cascade(_cfg(pregated_project, **cascade))
    st = _status(pregated_project / 'plm.sqlite3')
    assert st['p3'] == st['p7'] == STATUS['SKIPPED']
    assert st['p1'] == STATUS['DONE']                            # already ran: left alone
    expect_px = STATUS['SKIPPED'] if cascade.get('skip_unranked') else STATUS['NOOPT']
    assert st['px'] == expect_px


def test_skipped_source_rows_do_not_count_towards_keep_fraction_of_ranked(pregated_project):
    # 7 ranked (p0 p2 p4 p5 p6 p8 p9); 0.5 of them = 4; rows changed: p3 p7 (source
    # skipped) + p6 p8 p9 (rank)
    assert apply_cascade(_cfg(pregated_project, keep_fraction=0.5)) == (7, 4, 5)
    assert _claimable(pregated_project / 'plm.sqlite3') == {'p0', 'p2', 'p4', 'p5', 'px'}


def test_keep_denominator_ranked_is_the_default(pregated_project):
    apply_cascade(_cfg(pregated_project, keep_fraction=0.5))
    st = _status(pregated_project / 'plm.sqlite3')
    assert apply_cascade(_cfg(pregated_project, keep_fraction=0.5, keep_denominator='ranked'))[2] == 0
    assert _status(pregated_project / 'plm.sqlite3') == st


def test_keep_denominator_ranked_plus_skipped_keeps_the_ungated_budget(pregated_project):
    # 7 ranked + 3 source-skipped = 10; 0.5 of them = 5 ranked pairs kept
    assert apply_cascade(_cfg(pregated_project, keep_fraction=0.5,
                              keep_denominator='ranked_plus_skipped')) == (7, 5, 4)
    assert _claimable(pregated_project / 'plm.sqlite3') == {'p0', 'p2', 'p4', 'p5', 'p6', 'px'}
    # never more than the ranked population
    assert apply_cascade(_cfg(pregated_project, keep_fraction=1.0,
                              keep_denominator='ranked_plus_skipped'))[1] == 7


def test_keep_count_reproduces_the_gated_set_under_a_pre_gate(gated_project):
    """Headline policy on an ungated source: top 50 % of all scored pairs, then
    Neff/L >= 0.1. Under the pre-gate the same pairs are 'keep_count <that
    set's size>'; keep_fraction of the shrunken ranked population is not."""
    assert apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1))[1] == 3
    headline = _claimable(gated_project / 'plm.sqlite3')
    assert headline == {'p0', 'p2', 'p4', 'px'}
    _pregate_source(gated_project / 'gauss.sqlite3')
    assert apply_cascade(_cfg(gated_project, keep_count=3, min_neffoverL=0.1))[1] == 3
    assert _claimable(gated_project / 'plm.sqlite3') == headline
    # 0.5 of the 7 ranked pairs is 4, not 3: the fraction has lost its denominator
    assert apply_cascade(_cfg(gated_project, keep_fraction=0.5, min_neffoverL=0.1))[1] == 4
    assert _claimable(gated_project / 'plm.sqlite3') == headline | {'p5'}


def test_source_row_that_later_finishes_is_handed_back(pregated_project):
    apply_cascade(_cfg(pregated_project, min_neffoverL=0.1))
    assert _status(pregated_project / 'plm.sqlite3')['p3'] == STATUS['SKIPPED']
    # the source was re-run without its pre-gate: p3 is DONE and deep enough now
    conn = sqlite3.connect(pregated_project / 'gauss.sqlite3')
    conn.execute("UPDATE couplings SET status = ?, log_lr_raw = 7.0, neffoverL = 0.5 WHERE name = 'p3'",
                 (STATUS['DONE'],))
    conn.commit()
    conn.close()
    assert apply_cascade(_cfg(pregated_project, min_neffoverL=0.1))[2] == 1
    assert _status(pregated_project / 'plm.sqlite3')['p3'] == STATUS['NOOPT']


def test_executor_never_claims_source_skipped_pairs(pregated_project):
    from massivora.executors.couple import BaseCouplingExecutor
    cfg = {**_cfg(pregated_project, keep_count=3, min_neffoverL=0.1),
           'paths': {'monomers': 'msa', 'couplings': 'ecs'},
           'concatenate': {'downsample': {}}}
    executor = BaseCouplingExecutor(cfg)
    try:
        claimed = {row['name'] for row in executor._claim_batch('job', 100)}
    finally:
        executor.close_db_connection()
    assert claimed == {'p0', 'p2', 'p4', 'px'}

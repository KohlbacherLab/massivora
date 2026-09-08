import argparse
import logging
import multiprocessing as mp
import os
import signal
import sqlite3
import time

import psutil
from Bio import SeqIO

from massivora.alignment import TextAlignment
from massivora.config import load_project_and_system_config
from massivora.db import STATUS, connect_db_ro, connect_db_rw, quote_identifier
from massivora.massiveilance import compute_id_range
from massivora.utils import setup_logging, worker_id


def _claim_batch(cfg, job_id, batch_size, table_name='alignments', id_range=None):
    n = int(batch_size)
    if n <= 0:
        return []

    conn = connect_db_rw(cfg)

    # Set row factory for dict-like access (SQLite-specific)
    if hasattr(conn, 'row_factory'):
        conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    start_id = None
    end_id = None
    if id_range is not None:
        try:
            start_id, end_id = id_range
        except Exception:
            logging.error(f"Invalid id_range format: {id_range}, expected (start_id, end_id)")

    table = quote_identifier(table_name)

    try:
        start = float(time.time())
        where_range_sql = ""
        params = {
            'running': STATUS['RUNNING'],
            'now': float(time.time()),
            'job_id': str(job_id),
            'noopt': STATUS['NOOPT'],
            'pending': STATUS['PENDING'],
            'n': n,
        }

        if start_id is not None and end_id is not None:
            where_range_sql = " AND id >= :start_id AND id < :end_id"
            params['start_id'] = start_id
            params['end_id'] = end_id

        # For align job, status null is not claimable
        cursor.execute(
            f"""
            WITH to_claim AS (
                SELECT id, pid, status AS prev_status
                FROM {table}
                WHERE (status IS NOT NULL AND status IN (:noopt, :pending))
                  AND claimed_at IS NULL
                  {where_range_sql}
                ORDER BY id
                LIMIT :n
            )
            UPDATE {table}
            SET status = CASE
                    WHEN id IN (SELECT id FROM to_claim WHERE prev_status IS NULL OR prev_status = :noopt) THEN :running
                    ELSE status + 1
                END,
                job_id = :job_id,
                claimed_at = :now
            WHERE id IN (SELECT id FROM to_claim)
            RETURNING id, pid, status
            """,
            params,
        )
        rows = cursor.fetchall()
        conn.commit()
        logging.debug(f"job_id={job_id} claimed {len(rows)} alignment jobs in {time.time() - start:.3f}s")
        return rows
    except Exception as e:
        logging.error(f"Failed to claim alignment jobs, rolling back: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def _run_alignment(task, monomer_path, align_cfg):
    protein = task['pid']
    monomer_path = os.path.join(monomer_path, protein)
    if not os.path.exists(monomer_path):
        logging.error(f"Missing MSA directory for protein {protein} at expected path: {monomer_path}")
        return {'pid': protein, 'rc': 2, 'error': 'missing_msa_dir'}

    try:
        with open(os.path.join(monomer_path, f"{protein}.txt")) as f:
            record = SeqIO.read(f, 'swiss')
    except Exception as e:
        logging.error(f"Failed to load SwissProt record for protein {protein}: {e}")
        return {'pid': protein, 'rc': 2, 'error': f'load_swiss:{e}'}

    try:
        alignment = TextAlignment(record)
        output_prefix = os.path.join(monomer_path, alignment[0].id)

        params = {
            'jackhmmer_path': align_cfg.get('jackhmmer_binary'),
            'iterations': int(align_cfg.get('iterations', 5)),
            'output_prefix': output_prefix,
            'threads': int(align_cfg.get('per_job_cpu', 4)),
            'database': align_cfg.get('database'),
        }

        alignment.AnalogueSearch(threshold=align_cfg.get('analogue_threshold', 0.2), **params)
        align_nseqs = int(len(alignment))

        alignment.Filtering_MSA_Gap(align_cfg.get('row_gap_threshold', 0.5), align_cfg.get('col_gap_threshold', 0.5))
        alignment.Filtering_MSA_Invalid()
        alignment.Best_Reciprocal_Hit(align_cfg.get('paralogue_threshold', 0.9))
        length = int(len(alignment[0]))
        filtered_nseqs = int(len(alignment))

        alignment.To_Zarr(monomer_path)

        if not align_cfg.get('keep_original_msa', False):
            sto_file = [
                output_prefix + '.sto',
                output_prefix + '.domtblout',
                output_prefix + '.tblout',
            ]
            for f in sto_file:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

        return {
            'pid': protein,
            'rc': 0,
            'length': length,
            'align_nseqs': align_nseqs,
            'filtered_nseqs': filtered_nseqs,
        }
    except Exception as e:
        logging.error(f"Failed to run alignment for protein {protein}: {e}")
        return {'pid': protein, 'rc': 1, 'error': f'align:{e}'}


def _db_apply_results(cfg, done_updates, fail_updates, table_name='alignments'):
    if not done_updates and not fail_updates:
        return 0

    conn = connect_db_rw(cfg)
    cur = conn.cursor()
    table = quote_identifier(table_name)

    try:
        n = 0
        if done_updates:
            cur.executemany(
                f"""
                UPDATE {table}
                SET length = ?,
                    align_nseqs = ?,
                    filtered_nseqs = ?,
                    status = ?,
                    job_id = NULL,
                    claimed_at = NULL
                WHERE pid = ?
                """,
                [
                    (
                        int(x.get('length', 0) or 0),
                        int(x.get('align_nseqs', 0) or 0),
                        int(x.get('filtered_nseqs', 0) or 0),
                        STATUS['DONE'],
                        x['pid'],
                    )
                    for x in done_updates
                ],
            )
            n += len(done_updates)

        if fail_updates:
            cur.executemany(
                f"""
                UPDATE {table}
                SET status = MIN(status + 1, ?),
                    job_id = NULL,
                    claimed_at = NULL
                WHERE pid = ?
                """,
                [(STATUS['FAILED'], pid) for pid in fail_updates],
            )
            n += len(fail_updates)

        conn.commit()
        return n
    except Exception as e:
        logging.error(f"Failed to apply alignment results to DB, rolling back: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def _worker_loop(task_q, result_q, monomers_path, align_cfg):
    while True:
        try:
            task = task_q.get(timeout=1)
        except Exception:
            continue
        if task is None:
            return

        res = _run_alignment(task, monomer_path=monomers_path, align_cfg=align_cfg)
        result_q.put(res)


def coordinator_loop(
    cfg,
    job_id,
    task_q,
    result_q,
    claim_batch_size,
    table_name='alignments',
    id_range=None,
    commit_every_tasks=64,
    commit_every_seconds=300,
):
    inflight = 0
    done_recs = []
    fail_recs = []
    last_commit = float(time.time())
    terminating = False

    def flush(force=False):
        nonlocal done_recs, fail_recs, last_commit
        if not done_recs and not fail_recs:
            return 0
        if not force:
            if (len(done_recs) + len(fail_recs)) < int(commit_every_tasks) and (time.time() - last_commit) < float(commit_every_seconds):
                return 0
        n = _db_apply_results(cfg, done_recs, fail_recs, table_name)
        done_recs = []
        fail_recs = []
        last_commit = float(time.time())
        return n

    def _on_term(signum, frame):
        nonlocal terminating
        if terminating:
            return
        terminating = True
        logging.warning(f"Received signal {signum}; flushing DB now, then stopping new task dispatch...")
        try:
            flush(force=True)
        except Exception:
            logging.exception("Failed to flush DB during termination")

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    # Claim operations open/close their own connection.
    # (No lease-reclaim here; massiveilance handles that.)
    no_more_to_claim = False

    while True:
        if not terminating and not no_more_to_claim:
            target_inflight = int(claim_batch_size)
            while inflight < target_inflight:
                to_claim = max(0, target_inflight - inflight)
                if to_claim <= 0:
                    break
                rows = _claim_batch(cfg, job_id, to_claim, table_name, id_range)
                if not rows:
                    no_more_to_claim = True
                    break
                for r in rows:
                    task_q.put({k: r[k] for k in r.keys()})
                    inflight += 1

        if inflight == 0 and (terminating or no_more_to_claim):
            break

        try:
            res = result_q.get(timeout=1.0)
            inflight -= 1
            if int(res.get('rc', 1)) == 0:
                done_recs.append(res)
            else:
                fail_recs.append(res.get('pid'))
        except Exception:
            pass

        flush(force=False)

    while inflight > 0:
        try:
            res = result_q.get(timeout=5)
            inflight -= 1
            if int(res.get('rc', 1)) == 0:
                done_recs.append(res)
            else:
                fail_recs.append(res.get('pid'))
        except Exception:
            pass
        flush(force=False)

    flush(force=True)
    if terminating:
        raise SystemExit(143)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein alignment executor')
    parser.add_argument('--config', required=True, help='Project config YAML')
    args = parser.parse_args()

    cfg = load_project_and_system_config(args.config)
    setup_logging(cfg)

    project_path = cfg.get('project').get('project_path')
    if not project_path:
        raise SystemExit('Missing config: project.project_path')
    monomers_path = os.path.join(project_path, cfg.get('paths').get('monomers'))

    align_cfg = cfg.get('align')
    align_table = align_cfg.get('db_table', 'alignments')

    portion = float(align_cfg.get('portion', 1.0))
    portion_start = float(align_cfg.get('portion_start', 0.0))

    conn = connect_db_ro(cfg)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COALESCE(MAX(id), 0) FROM {quote_identifier(align_table)}")
    total = int(cursor.fetchone()[0] or 0)
    conn.close()

    per_job_cpu = int(align_cfg.get('per_job_cpu', 4))
    n_worker = max(1, int(psutil.cpu_count(logical=True) / per_job_cpu))
    claim_batch_size = align_cfg.get('claim_batch_size', 'auto')
    if claim_batch_size == 'auto':
        claim_batch_size = n_worker
    else:
        claim_batch_size = int(claim_batch_size)

    job_id = worker_id()

    task_q = mp.Queue()
    result_q = mp.Queue()

    workers = []
    for _ in range(int(n_worker)):
        p = mp.Process(
            target=_worker_loop,
            kwargs=dict(
                task_q=task_q,
                result_q=result_q,
                monomers_path=monomers_path,
                align_cfg=align_cfg,
            ),
        )
        p.start()
        workers.append(p)

    try:
        coordinator_loop(
            cfg=cfg,
            job_id=job_id,
            task_q=task_q,
            result_q=result_q,
            claim_batch_size=claim_batch_size,
            table_name=align_table,
            id_range=compute_id_range(portion, portion_start, total),
            commit_every_tasks=64,
            commit_every_seconds=300,
        )
    finally:
        for _ in workers:
            task_q.put(None)
        for p in workers:
            p.join()

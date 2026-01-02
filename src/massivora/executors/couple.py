import argparse
import multiprocessing as mp
import logging
import os
import sqlite3
import subprocess
import time

import psutil

from massivora.alignment import ZarrAlignment
from massivora.config import load_project_and_system_config
from massivora.db import connect_db_rw, STATUS
from massivora.logging_utils import setup_logging


def _worker_id():
    return f"{os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID')}"


def _claim_batch(db_path, claimed_by, batch_size):
    """Atomically claim up to batch_size coupling tasks in one SQL."""
    n = int(batch_size)
    if n <= 0:
        return []

    conn = connect_db_rw(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Status model:
        # - First ever run: NOOPT (-1) -> RUNNING (1)
        # - Retry run: PENDING (2) -> RETRYING (3) via +1
        start = float(time.time())
        cursor.execute(
            """
            WITH to_claim AS (
                SELECT id, status AS prev_status
                FROM couplings
                WHERE status IN (:noopt, :pending)
                ORDER BY id
                LIMIT :n
            )
            UPDATE couplings
            SET status = CASE
                    WHEN id IN (SELECT id FROM to_claim WHERE prev_status = :noopt) THEN :running
                    ELSE status + 1
                END,
                job_id = :job_id,
                claimed_at = :now
            WHERE id IN (SELECT id FROM to_claim)
            RETURNING *
            """,
            {
                'running': STATUS['RUNNING'],
                'now': float(time.time()),
                'job_id': str(claimed_by),
                'noopt': STATUS['NOOPT'],
                'pending': STATUS['PENDING'],
                'n': n,
            },
        )
        rows = cursor.fetchall()
        conn.commit()
        logging.debug(f"Worker {claimed_by} claimed {len(rows)} coupling jobs in {time.time() - start:.3f} seconds")
        return rows
    finally:
        conn.close()


def run_plmc_row(row, alignment_dir, db_path, plmc_binary, threads, lambda_h=None, lambda_J=None, lambda_g=None):
    pid1, pid2 = row['pid1'], row['pid2']
    index = row['id']

    concat_filename = os.path.join(alignment_dir, f"{pid1}-{pid2}.a2m")
    zarrfile = os.path.join(alignment_dir, f"{pid1}-{pid2}")
    zarralign = ZarrAlignment(zarrfile)
    weightfile = os.path.join(alignment_dir, f"{pid1}-{pid2}.weights")
    with open(weightfile, 'w') as f:
        for weight in zarralign.weights:
            f.write(f"{weight:.3f}\n")

    le_value = None
    if lambda_J is not None:
        le_value = str(float(lambda_J) * (row['length'] - 1) * 20)

    cmd = [
        plmc_binary,
        '-c', os.path.join(alignment_dir, f"{pid1}-{pid2}.txt"),
        '-o', '/dev/null',
        '-f', '0',
        '-g',
        '-m', '100',
        '-t', '0.2',
        '-w', weightfile
    ]
    if lambda_h is not None:
        cmd += ['-lh', str(lambda_h)]
    if le_value is not None:
        cmd += ['-le', le_value]
    cmd += ['-n', str(threads), concat_filename]
    logging.debug(f"Running PLMC for coupling id {index} ({pid1}-{pid2}): {' '.join(cmd)}")
    rv = subprocess.run(cmd)

    # Status progression (one auto-retry, encoded in integer value):
    # RUNNING (1)  : first attempt in progress
    # PENDING (2)  : first attempt failed; waiting to rerun
    # RETRYING (3) : second attempt in progress
    # FAILED (4)   : second attempt failed
    #
    # Rules requested:
    # - First attempt is explicitly set when first run starts.
    # - Later transitions use += 1 semantics.

    conn = connect_db_rw(db_path)
    cursor = conn.cursor()
    try:
        if rv.returncode == 0:
            try:
                os.remove(concat_filename)
                os.remove(weightfile)
            except OSError:
                pass
            cursor.execute(
                "UPDATE couplings SET status = ?, claimed_at = NULL WHERE id = ?",
                (STATUS['DONE'], int(index)),
            )
        else:
            logging.error(f"PLMC failed for coupling id {index} ({pid1}-{pid2}) with return code {rv.returncode}")

            # On failure, advance status by +1:
            # RUNNING -> PENDING; RETRYING -> FAILED
            cursor.execute(
                """
                UPDATE couplings
                SET status = MIN(status + 1, :failed),
                    claimed_at = NULL
                WHERE id = :id
                """,
                {
                    'failed': STATUS['FAILED'],
                    'id': int(index),
                },
            )
        conn.commit()
    finally:
        conn.close()


def worker_loop(alignment_dir, db_path, plmc_binary, threads, lambda_h=None, lambda_J=None, claim_batch_size=4):
    wid = _worker_id()

    while True:
        rows = _claim_batch(db_path, wid, claim_batch_size)
        if not rows:
            return

        for row in rows:
            run_plmc_row(
                row,
                alignment_dir=alignment_dir,
                db_path=db_path,
                plmc_binary=plmc_binary,
                threads=threads,
                lambda_h=lambda_h,
                lambda_J=lambda_J,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coupling calculation executor')
    parser.add_argument('--config', required=True, help='Project config YAML')
    # parser.add_argument('jobs', type=str, nargs='?', default=None, help='Optional job list file')

    args = parser.parse_args()

    cfg = load_project_and_system_config(args.config)
    setup_logging(cfg)

    project_path = cfg.get('project').get('project_path')
    if not project_path:
        raise SystemExit('Missing config: project.project_path')

    job_db = os.path.join(project_path, cfg.get('paths').get('job_db'))
    if not job_db:
        raise SystemExit('Missing config: paths.job_db')

    alignment_dir = os.path.join(project_path, cfg.get('paths').get('couplings'))

    coupling_cfg = cfg.get('coupling')
    method = coupling_cfg.get('method', 'plm')
    per_job_cpu = int(coupling_cfg.get('per_job_cpu', 8))
    p_jobs = max(1, int(psutil.cpu_count(logical=False) / per_job_cpu))
    plmc_binary = coupling_cfg.get('plmc_binary')
    if not plmc_binary:
        raise SystemExit('Missing config: coupling.plmc_binary')

    lambda_h = coupling_cfg.get('lambda_h')
    lambda_J = coupling_cfg.get('lambda_J')

    claim_batch_size = int(coupling_cfg.get('claim_batch_size', 4))

    procs = []
    for _ in range(p_jobs):
        p = mp.Process(
            target=worker_loop,
            kwargs=dict(
                alignment_dir=alignment_dir,
                db_path=job_db,
                plmc_binary=plmc_binary,
                threads=per_job_cpu,
                lambda_h=lambda_h,
                lambda_J=lambda_J,
                claim_batch_size=claim_batch_size,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

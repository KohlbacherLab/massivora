import argparse
import logging
import os
import sqlite3
import subprocess
import socket
import time
from datetime import datetime
from multiprocessing import shared_memory

import zarr

from massivora.config import load_project_and_system_config
from massivora.db import (STATUS, connect_db, connect_db_ro, get_hashed_file_path,
                          get_table_names, quote_identifier)
from massivora.job_loader import SlurmJobLoader
from massivora.utils import SHM_PREFIX, compute_id_range, setup_logging

WORKER_NAME = socket.gethostname()

# Local runs: give up (and let massiveilance remove itself from crontab) after
# this many consecutive ticks with no task progress while work still remains.
MAX_STALL_TICKS = 5

def _get_table_names(cfg):
    """Get table names from config."""
    return get_table_names(cfg)


def _slurm_queuing_job_ids():
    try:
        rv = subprocess.run(
            ['squeue', '-h', '-u', os.environ.get('USER', ''), '-o', '%i'],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return set()

    if rv.returncode != 0:
        return set()

    alive = (rv.stdout or '').splitlines()
    return alive

def _too_many_failures(cfg, command, mode):
    """Decide whether the run is hopeless enough to stop monitoring."""
    if mode == 'batch':
        return _too_many_failures_slurm(cfg)
    return _too_many_failures_local(cfg, command)


def _too_many_failures_slurm(cfg):
    """
    Check if the last two batches all have FAILED status.
    This function only applies to the SLURM backend.

    Parameters
    ----------
    `cfg` — dict
        Configuration dict

    Returns
    -------
    `bool`
        True if the last two batches all failed, False otherwise
    """
    batch = cfg.get('batch')
    batch_size = int(batch.get('maximum_nodes', 1))
    two_batches = batch_size * 2

    conn = connect_db_ro(cfg)

    # Set row factory for dict-like access (SQLite-specific)
    if hasattr(conn, 'row_factory'):
        conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        # Get the last two batches worth of jobs from the jobs table ordered by job_id DESC
        rows = cursor.execute(
            "SELECT status FROM jobs ORDER BY job_id DESC LIMIT ?",
            (two_batches,)
        ).fetchall()

        # If we don't have enough jobs yet, don't fail
        if len(rows) < two_batches:
            return False

        # Check if all of them have status 'FAILED'
        return all(row['status'] == 'FAILED' for row in rows)
    finally:
        conn.close()


def _save_monitor_progress(conn, stage, done_count, stall_ticks):
    """Upsert the local-run progress record for ``stage`` (creates table lazily)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS monitor_progress (
            stage TEXT PRIMARY KEY,
            done_count INTEGER NOT NULL,
            stall_ticks INTEGER NOT NULL,
            updated_at REAL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO monitor_progress (stage, done_count, stall_ticks, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(stage) DO UPDATE SET
            done_count = excluded.done_count,
            stall_ticks = excluded.stall_ticks,
            updated_at = excluded.updated_at
        """,
        (stage, int(done_count), int(stall_ticks), time.time()),
    )


def _too_many_failures_local(cfg, command):
    """
    Local-run analogue of the SLURM too-many-failures guard.
    """
    table_names = _get_table_names(cfg)
    if command == 'align':
        table = quote_identifier(table_names['alignments'])
        stage_cfg = cfg.get('align') or {}
    elif command == 'couple':
        table = quote_identifier(table_names['couplings'])
        stage_cfg = cfg.get('couple') or {}
    else:
        return False

    try:
        conn = connect_db(cfg)
    except Exception as e:
        logging.warning(f"massiveilance on {WORKER_NAME}: progress check skipped; DB open failed: {e}")
        return False

    if hasattr(conn, 'row_factory'):
        conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        # DONE count and whether any workable (non-terminal) task remains, within
        # the configured slice of this stage's table.
        portion = float(stage_cfg.get('portion', 1.0))
        portion_start = float(stage_cfg.get('portion_start', 0.0))
        cursor.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}")
        total = int(cursor.fetchone()[0] or 0)
        idx_start, idx_end = compute_id_range(portion, portion_start, total)

        done_now = int(cursor.execute(
            f"SELECT COUNT(*) FROM {table} WHERE status = ? AND id >= ? AND id < ?",
            (STATUS['DONE'], idx_start, idx_end),
        ).fetchone()[0] or 0)

        workable = cursor.execute(
            f"SELECT 1 FROM {table} WHERE status NOT IN (?, ?) AND id >= ? AND id < ? LIMIT 1",
            (STATUS['DONE'], STATUS['FAILED'], idx_start, idx_end),
        ).fetchone()

        prev = cursor.execute(
            "SELECT done_count, stall_ticks FROM monitor_progress WHERE stage = ?",
            (command,),
        ).fetchone() if _monitor_progress_exists(conn) else None

        # Nothing workable left: completion handles removal. Reset the stall
        # counter so a fresh re-run isn't immediately judged as stuck.
        if workable is None:
            _save_monitor_progress(conn, command, done_now, 0)
            conn.commit()
            return False

        # First observation, or progress since last tick: (re)seed and wait.
        if prev is None or done_now > int(prev['done_count']):
            _save_monitor_progress(conn, command, done_now, 0)
            conn.commit()
            return False

        # No progress this tick: advance the stall counter.
        stall = int(prev['stall_ticks']) + 1
        _save_monitor_progress(conn, command, done_now, stall)
        conn.commit()

        if stall >= MAX_STALL_TICKS:
            logging.warning(
                f"massiveilance on {WORKER_NAME} {command}: no progress for {stall} consecutive "
                f"runs (DONE stuck at {done_now}, workable tasks remain); treating run as failed."
            )
            return True
        return False
    except Exception as e:
        logging.warning(f"massiveilance on {WORKER_NAME}: progress check failed: {e}")
        return False
    finally:
        conn.close()


def _monitor_progress_exists(conn):
    """True if the local-run progress table has been created in this DB."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='monitor_progress'"
    ).fetchone()
    return row is not None


def _project_is_complete(cfg, command):
    """
    Check if the project stage is complete.

    Parameters
    ----------
    `cfg` — dict
        Configuration dict
    `command` — str
        Either 'align' or 'couple' to determine which table to check

    Returns
    -------
    `bool`
        True if all jobs in the specified stage are DONE or FAILED, False otherwise
    """
    table_names = _get_table_names(cfg)

    if command == 'align':
        table = quote_identifier(table_names['alignments'])
        stage_cfg = cfg.get('align')
    elif command == 'couple':
        table = quote_identifier(table_names['couplings'])
        stage_cfg = cfg.get('couple')
    else:
        logging.error(f"massiveilance on {WORKER_NAME}: invalid command {command} for project completion check")
        return True

    conn = connect_db_ro(cfg)

    # Set row factory for dict-like access (SQLite-specific)
    if hasattr(conn, 'row_factory'):
        conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        # Is finished if there are no rows outside DONE/FAILED in the slice
        portion = float(stage_cfg.get('portion', 1.0))
        portion_start = float(stage_cfg.get('portion_start', 0.0))
        cursor.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}")
        total = int(cursor.fetchone()[0] or 0)

        idx_start, idx_end = compute_id_range(portion, portion_start, total)

        where = ["status NOT IN (?, ?)", "id >= ? AND id < ?"]
        params = [STATUS['DONE'], STATUS['FAILED'], idx_start, idx_end]

        row = cursor.execute(
            f"SELECT 1 FROM {table} WHERE {' AND '.join(where)} LIMIT 1",
            params,
        ).fetchone()

        if row is not None:
            return False

        return True
    finally:
        conn.close()

def _get_crontab():
    proc = subprocess.run(
        ["crontab", "-l"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )
    return proc.stdout.splitlines()


def _write_crontab(lines):
    content = "\n".join(lines) + "\n"
    subprocess.run(
        ["crontab", "-"],
        input=content,
        text=True,
        check=True
    )


def ensure_cron_entry(cron_line):
    lines = _get_crontab()

    if cron_line in lines:
        return False

    lines.append(cron_line)
    _write_crontab(lines)
    return True


def remove_cron_entry(cron_line):
    lines = _get_crontab()

    if cron_line not in lines:
        return False

    lines = [line for line in lines if line != cron_line]
    _write_crontab(lines)
    return True


def _coupling_result_info(output_dir, couple_table, pair_name, file_hash):
    """Read a finished pair's coupling result from zarr, or ``None`` if absent."""
    filename, _ = get_hashed_file_path(pair_name, file_hash, output_dir)
    if not os.path.exists(filename):
        return None
    try:
        grp = zarr.open_group(store=filename, mode='r')
        arrays = set(grp.array_keys())
        if couple_table not in arrays:
            return None
        info = {'length': int(grp[couple_table].shape[0]), 'number': None, 'effnumber': None}
        if 'align' in arrays:
            align = grp['align']
            info['number'] = int(align.shape[0])
            beff = align.attrs.get('Beff')
            info['effnumber'] = None if beff is None else float(beff)
        return info
    except Exception:
        return None


def _alignment_result_info(monomers_dir, pid):
    """Read a finished protein's alignment result from zarr, or ``None`` if absent."""
    store = os.path.join(monomers_dir, pid)
    if not os.path.exists(store):
        return None
    try:
        grp = zarr.open_group(store=store, mode='r')
        if 'align' not in set(grp.array_keys()):
            return None
        shape = grp['align'].shape
        return {'length': int(shape[1]), 'filtered_nseqs': int(shape[0])}
    except Exception:
        return None


def fix_zombie_tasks(cfg, command):
    """
    Fix tasks that are RUNNING/RETRYING but whose SLURM job_id no longer exists. Always clears claimed_at/job_id.

    Rule:
    - RETRYING -> PENDING
    - RUNNING  -> NOOPT

    Parameters
    ----------
    `cfg` — dict
        Configuration dict

    Returns
    -------
    `int`
        The number of fixed jobs
    """
    table_names = _get_table_names(cfg)
    if command == 'align':
        task_table = quote_identifier(table_names['alignments'])
        select_columns = 'id, pid, status, job_id, length, align_nseqs, filtered_nseqs'
        is_coupling_task = False
    elif command == 'couple':
        task_table = quote_identifier(table_names['couplings'])
        select_columns = 'id, name, pid1, pid2, status, job_id, file_hash, number, length, effnumber'
        is_coupling_task = True
    else:
        logging.error(f"massiveilance on {WORKER_NAME}: invalid command {command} for zombie task recovery")
        return 0

    # Resolve the output locations used to tell a finished task from a dead one
    # (the zarr the executor writes on success), exactly as the executors do.
    project_path = cfg.get('project').get('project_path')
    couplings_output_dir = None
    couple_table_name = None
    monomers_dir = None
    if is_coupling_task:
        couplings_output_dir = os.path.join(project_path, cfg.get('paths').get('couplings'))
        couple_table_name = table_names['couplings']
    else:
        monomers_dir = os.path.join(project_path, cfg.get('paths').get('monomers'))

    conn = connect_db(cfg)

    # Set row factory for dict-like access (SQLite-specific)
    if hasattr(conn, 'row_factory'):
        conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    rows = cursor.execute(
        f"""
                SELECT {select_columns}
                FROM {task_table}
        WHERE job_id IS NOT NULL
          AND claimed_at IS NOT NULL
          AND status IN (?, ?)
        """,
        (STATUS['RUNNING'], STATUS['RETRYING']),
    ).fetchall()

    alive = set(_slurm_queuing_job_ids())

    job_ids_to_fix = set([str(r['job_id']) for r in rows if r['job_id']]) - alive

    fixed = 0
    try:
        for r in rows:
            jid = str(r['job_id'])
            if jid not in job_ids_to_fix:
                continue

            old_status = r['status']

            # A dead job whose executor already wrote its result zarr actually
            # finished; only its DB status update was lost. Mark it DONE instead
            # of re-queuing it
            if is_coupling_task: # TODO: check the probabilities records as well
                result = _coupling_result_info(
                    couplings_output_dir, couple_table_name, r['name'], r['file_hash']
                )
            else:
                result = _alignment_result_info(monomers_dir, r['pid'])

            set_clauses = ["status = ?", "claimed_at = NULL", "job_id = NULL"]
            values = []
            if result is not None:
                new_status = STATUS['DONE']
                # Backfill any metric the executor never got to write (its
                # completion update was lost), leaving present values untouched.
                for col, val in result.items():
                    if val is not None and r[col] is None:
                        set_clauses.append(f"{col} = ?")
                        values.append(val)
            elif old_status == STATUS['RETRYING']:
                new_status = STATUS['PENDING']
            else:
                new_status = STATUS['NOOPT']

            # fix db record
            values.insert(0, new_status)
            values.append(r['id'])
            cursor.execute(
                f"UPDATE {task_table} SET {', '.join(set_clauses)} WHERE id = ?",
                values,
            )

            if is_coupling_task:
                # fix shared memory record (names carry the SHM_PREFIX applied
                # at allocation time in the coupling executor)
                shm_name = f"{SHM_PREFIX}{r['pid1']}-{r['pid2']}"
                try:
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass
                try:
                    shm = shared_memory.SharedMemory(name=shm_name+'_J')
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

            fixed += 1

        conn.commit()
    finally:
        conn.close()
    return fixed

def update_slurm_job_status(cfg, command):
    """
    Update the SLURM job status in the database. This function uses `sacct` to
    get the current status of jobs and updates the `jobs` table accordingly.
    It will put the exact SLURM state string to the `status` column.

    Parameters
    ----------
    `cfg` — dict
        Configuration dict
    `command` — str
        Either 'align' or 'couple' to determine which task table to update
    """
    table_names = _get_table_names(cfg)
    if command == 'align':
        task_table = quote_identifier(table_names['alignments'])
    elif command == 'couple':
        task_table = quote_identifier(table_names['couplings'])
    else:
        logging.error(f"massiveilance on {WORKER_NAME}: invalid command {command} for SLURM job status update")
        return

    conn = connect_db(cfg)
    cursor = conn.cursor()
    try:
        rows = cursor.execute("""SELECT job_id, status FROM jobs WHERE status IN ('RUNNING', 'PENDING')""").fetchall()

        job_ids = sorted({str(r[0]) for r in rows if r[0] is not None})
        if not job_ids:
            return

        try:
            rv = subprocess.run(
                [
                    "sacct",
                    "-n",
                    "-P",
                    "-o",
                    "JobIDRaw,State",
                    "-j",
                    ",".join(job_ids),
                ],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return

        if rv.returncode != 0:
            return

        sacct_state_by_jobid = {}
        for ln in (rv.stdout or "").splitlines():
            parts = ln.split("|", 1)
            if len(parts) != 2:
                continue
            job_id, current_state = parts[0].strip(), parts[1].strip()
            sacct_state_by_jobid[job_id] = current_state.split()[0].split("+", 1)[0].strip()

        for job_id, old_status in rows:
            current_state = sacct_state_by_jobid.get(job_id)
            if old_status == current_state:
                continue
            # If state has changed
            cursor.execute(
                "UPDATE jobs SET status = ? WHERE job_id = ?",
                (current_state, job_id),
            )
            # If the job didn't complete successfully, update the task status
            if current_state not in ('RUNNING', 'PENDING', 'COMPLETED'):
                cursor.execute(
                    f"""
                    UPDATE {task_table}
                    SET status = MIN(status + 1, ?),
                        claimed_at = NULL,
                        job_id = NULL
                    WHERE job_id = ?
                      AND status != ?
                    """,
                    (STATUS['FAILED'], job_id, STATUS['DONE']),
                )

        conn.commit()
    finally:
        conn.close()

def maybe_resubmit(cfg, config_path, stage, mode):
    """
    Resubmit a job if needed.

    Parameters
    ----------
    `cfg` — dict
        Configuration dict
    `config_path` — str
        Path to the project config file
    `stage` — str
        Either 'align' or 'couple' to determine which stage to resubmit
    `mode` — str
        The massivora command being monitored: 'batch' (SLURM) or 'run' (local)

    Returns
    -------
    `bool`
        True if a job was resubmitted, False otherwise
    """
    if mode == 'batch':
        # For SLURM runs, if running jobs are below maximum_nodes,
        # submit a new coordinator job.
        batch = cfg.get('batch')
        maximum_nodes = int(batch.get('maximum_nodes', 1))

        active_jobs = len(_slurm_queuing_job_ids())

        if active_jobs >= maximum_nodes:
            return False

        # Re-submit the sbatch scripts that the initial `massivora batch` run already generated
        loader = SlurmJobLoader(project_cfg=config_path)
        return loader.resubmit(stage) > 0
    else:
        # Else monitor the coordinator process,
        # and restart it if it's dead.

        return False


def main(argv=None):
    p = argparse.ArgumentParser(prog='massiveilance')
    p.add_argument('mode', choices=['run', 'batch'], help='Massivora command being monitored (run=local, batch=SLURM)')
    p.add_argument('command', choices=['align', 'couple'], help='Stage being monitored (align or couple)')
    p.add_argument('config', help='Project config YAML (same as `massivora <mode> <command> <config>`).')
    args = p.parse_args(argv)

    cfg = load_project_and_system_config(args.config)
    project_path = cfg.get('project').get('project_path')

    setup_logging(cfg)
    logging.info(f'massiveilance {args.command} {args.config}: Regular massiveilance run started on node {socket.gethostname()}.')

    # 0) update job status
    if args.mode == 'batch':
        update_slurm_job_status(cfg, args.command)
        logging.info(f"massiveilance on {WORKER_NAME} {args.command} {args.config}: job status in database has been updated")

    # 1) completion check
    if _project_is_complete(cfg, args.command) or _too_many_failures(cfg, args.command, args.mode):
        jobs = _get_crontab()
        if jobs:
            for job in jobs:
                if ('massiveilance' in job and f'{args.mode} {args.command}' in job
                        and os.path.abspath(args.config) in job):
                    if remove_cron_entry(job):
                        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Regular massiveilance run started.',
                              f"\n\nProject {args.command} {args.config} is complete; removed itself from crontab.")
                        logging.info(f"massiveilance on {WORKER_NAME} {args.command} {args.config}: project is complete; removed itself from crontab")
            return 0
        logging.info(f"massiveilance on {WORKER_NAME} {args.command} {args.config}: project is complete, project not in crontab")
        return 0

    # 2) fix orphaned tasks
    fixed = fix_zombie_tasks(cfg, args.command)
    if fixed:
        logging.info(f"massiveilance on {WORKER_NAME} {args.command} {args.config}: fixed {fixed} zombie {args.command} tasks")

    # 3) maybe resubmit
    if maybe_resubmit(cfg, os.path.abspath(args.config), args.command, args.mode):
        logging.info(f"massiveilance on {WORKER_NAME} {args.command} {args.config}: resubmitted massivora job")
        return 0

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

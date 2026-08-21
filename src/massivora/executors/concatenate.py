import argparse
import logging
import multiprocessing as mp
import os
import signal
import sqlite3
import time

import numpy as np
import psutil

from massivora.alignment import BinaryAlignment
from massivora.config import load_project_and_system_config
from massivora.db import (compute_file_hash, connect_db_rw,
                          get_hashed_file_path, get_table_names,
                          quote_identifier)
from massivora.utils import (gpu_is_available, setup_logging,
                             visible_gpu_devices, worker_id)


def _claim_batch(cfg, job_id, batch_size, table_name='couplings'):
    n = int(batch_size)
    if n <= 0:
        return []

    conn = connect_db_rw(cfg)

    # Set row factory for dict-like access (SQLite-specific)
    if hasattr(conn, 'row_factory'):
        conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    table = quote_identifier(table_name)

    try:
        start = float(time.time())
        params = {
            'n': n,
        }

        # A pair waits for concatenation while `number` is NULL. Claiming writes
        # 0 into it, so no other node picks the same pair up; the real sequence
        # count replaces the 0 once the alignment is written out, and a pair
        # that failed simply stays at 0.
        cursor.execute(
            f"""
            WITH to_claim AS (
                SELECT id
                    FROM {table}
                    WHERE number IS NULL
                    ORDER BY id
                    LIMIT :n
            )
            UPDATE {table}
            SET number = 0
            WHERE id IN (SELECT id FROM to_claim)
            RETURNING name, pid1, pid2
            """,
            params,
        )
        rows = cursor.fetchall()
        conn.commit()
        logging.debug(f"job_id={job_id} claimed {len(rows)} concatenating jobs in {time.time() - start:.3f}s")
        return rows
    except Exception as e:
        logging.error(f"Failed to claim concatenating jobs, rolling back: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def _run_concatenate(task, monomers_path, output_dir, downsample=False,
                     downsample_to=30000, reweight=True, reweighting_threshold=0.8,
                     col_gap_threshold=0.5, n_threads=0, gpu_reweight=None):
    """
    Concatenate one pair of monomer alignments and write it to the output tree.

    Reweighting runs in this process on `n_threads` threads, unless
    `gpu_reweight` is given: that callable hands the matrix to the GPU pool and
    returns the weights, leaving everything else here on the CPU.

    Returns
    -------
    `dict` or `None`
        The record to write back to the database, or `None` if the pair could
        not be concatenated. A failed pair keeps the 0 written into `number`
        when it was claimed, so it is neither retried nor taken for a finished
        one.
    """
    name = task['name']
    pid1 = task['pid1']
    pid2 = task['pid2']

    try:
        align1 = BinaryAlignment(os.path.join(monomers_path, pid1))
        align2 = BinaryAlignment(os.path.join(monomers_path, pid2))
    except Exception as e:
        logging.error(f"Failed to load monomer alignments for pair '{name}': {e}")
        return None

    try:
        concatenated_align = align1 + align2
        del align1, align2

        if downsample:
            concatenated_align.Downsample_Randomly(to=downsample_to)
        B, N = concatenated_align.matrix.shape
        if B == 0:
            logging.error(f"Pair '{name}' shares no species between {pid1} and {pid2}")
            return None

        if reweight:
            if gpu_reweight is None:
                concatenated_align.Reweight_Sequence(x=reweighting_threshold,
                                                     n_threads=n_threads)
            else:
                w = gpu_reweight(name, concatenated_align.matrix, reweighting_threshold)
                if w is None:
                    return None
                # Same two attributes Reweight_Sequence would have set.
                concatenated_align.weights = w.tolist()
                concatenated_align.Beff = float(w.sum())
            Beff = concatenated_align.Beff
        else:
            Beff = B
        concatenated_align.Mask_Gap_Columns(col_gap_threshold)
    except Exception as e:
        logging.error(f"Failed to concatenate pair '{name}': {e}")
        return None

    file_hash = compute_file_hash(name)
    filename, _ = get_hashed_file_path(name, file_hash, output_dir)

    try:
        concatenated_align.To_Zarr(filename, overwrite=True)
    except Exception as e:
        logging.error(f"Failed to save concatenated alignment {name}: {e}")
        return None

    return {
        'effnumber': Beff,
        'number': B,
        'length': N,
        'file_hash': file_hash,
        'name': name,
    }


def _db_apply_results(cfg, done_updates, table_name='couplings'):
    if not done_updates:
        return 0

    conn = connect_db_rw(cfg)
    cur = conn.cursor()
    table = quote_identifier(table_name)

    try:
        cur.executemany(
            f"""
            UPDATE {table}
            SET effnumber = ?,
                number = ?,
                length = ?,
                file_hash = ?
            WHERE name = ?
            """,
            [
                (
                    int(round(float(x.get('effnumber') or 0))),
                    int(x.get('number')),
                    int(x.get('length')),
                    x.get('file_hash'),
                    x.get('name'),
                )
                for x in done_updates
            ],
        )
        conn.commit()
        return len(done_updates)
    except Exception as e:
        logging.error(f"Failed to apply alignment results to DB, rolling back: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def _worker_loop(task_q, result_q, cfg, monomers_path, output_dir, concat_opts,
                 gpu_q=None, reply_q=None, worker_idx=0, shutdown_evt=None):
    """
    One concatenation worker: pull a pair off the queue, concatenate and
    reweight it, push the database record back.

    This process never touches the GPU. On the GPU path it sends the
    concatenated matrix to the reweighting pool and waits for the weights, so
    loading, concatenating, column filtering and writing all stay on this core.

    Parameters
    ----------
    `task_q` — multiprocessing.Queue
        Pairs to process; a `None` item ends the worker.
    `result_q` — multiprocessing.Queue
        Records handed back to the coordinator, `None` for a failed pair.
    `cfg` — dict
        Configuration dict, used to reattach logging in a spawned worker.
    `monomers_path` — str
        Directory holding the per-protein alignments.
    `output_dir` — str
        Root of the hashed output tree for concatenated alignments.
    `concat_opts` — dict
        Keyword arguments forwarded to `_run_concatenate`.
    `gpu_q` — multiprocessing.Queue (optional)
        Reweighting requests for the GPU pool, or `None` to reweight in this
        process on the CPU (default: `None`)
    `reply_q` — multiprocessing.Queue (optional)
        This worker's own queue for weights coming back (default: `None`)
    `worker_idx` — int (optional)
        Index of this worker in `reply_qs`, sent with every request so the GPU
        pool knows where to answer (default: `0`)
    `shutdown_evt` — multiprocessing.Event (optional)
        Set by the coordinator on the way out, to release a worker that is
        waiting on a GPU reply that will never come (default: `None`)
    """
    # The coordinator drives shutdown through the None sentinel; ignoring
    # SIGINT keeps a local Ctrl-C from tearing a worker down mid-write.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # A spawned worker starts with default logging configuration.
    setup_logging(cfg)

    gpu_reweight = None
    if gpu_q is not None:
        def gpu_reweight(name, matrix, threshold):
            """Hand one matrix to the GPU pool and wait for its weights."""
            gpu_q.put({'worker_idx': worker_idx, 'name': name,
                       'matrix': matrix, 'threshold': threshold})
            while True:
                try:
                    return reply_q.get(timeout=5)
                except Exception:
                    # A large alignment can hold the GPU for a long while, so
                    # only give up once the run is being torn down.
                    if shutdown_evt is not None and shutdown_evt.is_set():
                        logging.error(f"Gave up on '{name}': shutting down while waiting for the GPU")
                        return None

    while True:
        # Set once the coordinator is tearing down: stop taking new pairs
        # rather than starting work whose GPU pool may already be gone.
        if shutdown_evt is not None and shutdown_evt.is_set():
            return
        try:
            task = task_q.get(timeout=1)
        except Exception:
            continue
        if task is None:
            return

        try:
            res = _run_concatenate(task, monomers_path=monomers_path,
                                   output_dir=output_dir,
                                   gpu_reweight=gpu_reweight, **concat_opts)
        except Exception:
            # Never let a single bad pair kill the worker: the coordinator
            # counts one result per dispatched task.
            logging.exception(f"Unhandled error while concatenating '{task.get('name')}'")
            res = None
        result_q.put(res)


def _gpu_worker_loop(gpu_q, reply_qs, cfg, device_id):
    """
    Reweighting service for a single GPU.

    This process does nothing but drive its device. Concatenation, column
    filtering and the zarr write stay with the concatenation workers, so the
    GPU is never held up by work that does not need it.

    Parameters
    ----------
    `gpu_q` — multiprocessing.Queue
        Requests from the concatenation workers; a `None` item ends the worker.
    `reply_qs` — list
        One queue per concatenation worker, indexed by a request's `worker_idx`.
    `cfg` — dict
        Configuration dict, used to reattach logging in the spawned worker.
    `device_id` — str
        GPU this process binds to.
    """
    # Set before anything in this process can initialise CUDA. The pool is
    # started with the 'spawn' method so this lands on a clean interpreter.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    setup_logging(cfg)
    logging.info(f"Reweighting worker {os.getpid()} pinned to GPU {device_id}")

    while True:
        try:
            req = gpu_q.get(timeout=1)
        except Exception:
            continue
        if req is None:
            return

        weights = None
        try:
            # Go through the alignment's own CUDA path rather than repeating
            # the kernel launch here; reweighting needs nothing but the matrix.
            shim = BinaryAlignment()
            shim.matrix = req['matrix']
            shim.Reweight_Sequence(x=req['threshold'], use_GPU=True)
            weights = np.asarray(shim.weights, dtype=np.float64)
            del shim
        except Exception:
            logging.exception(f"GPU {device_id} failed to reweight pair '{req.get('name')}'")

        # Always answer: a worker without its weights would wait forever.
        reply_qs[req['worker_idx']].put(weights)


def _drain_and_join(procs, drain_qs=(), timeout=30):
    """
    Wait for `procs` to exit, keeping `drain_qs` empty meanwhile.

    A process that has put an item on a queue nobody reads cannot exit: its
    feeder thread stays blocked on a full pipe. That happens when the pool on
    the other side died, so the queues it was talking to have to be emptied
    while waiting. Anything still alive at the deadline is terminated.
    """
    deadline = time.time() + float(timeout)
    for p in procs:
        while p.is_alive() and time.time() < deadline:
            drained = False
            for q in drain_qs:
                try:
                    q.get(timeout=0.1)
                    drained = True
                except Exception:
                    pass
            if not drained:
                p.join(timeout=0.2)
        if p.is_alive():
            logging.warning(f"Worker {p.pid} did not stop on its own; terminating it")
            p.terminate()
            p.join(timeout=10)


def coordinator_loop(
    cfg,
    job_id,
    task_q,
    result_q,
    claim_batch_size,
    table_name='couplings',
    workers=None,
    commit_every_tasks=64,
    commit_every_seconds=300,
):
    inflight = 0
    done_recs = []
    last_commit = float(time.time())
    terminating = False

    def flush(force=False):
        nonlocal done_recs, last_commit
        if not done_recs:
            return 0
        if not force:
            if len(done_recs) < int(commit_every_tasks) and (time.time() - last_commit) < float(commit_every_seconds):
                return 0
        n = _db_apply_results(cfg, done_recs, table_name)
        done_recs = []
        last_commit = float(time.time())
        return n

    def collect(timeout):
        """Take one result off the queue. True if one was collected."""
        nonlocal inflight
        try:
            res = result_q.get(timeout=timeout)
        except Exception:
            return False
        inflight -= 1
        # A failed pair needs no update: it keeps the 0 written at claim time.
        if res:
            done_recs.append(res)
        return True

    def workers_gone():
        """True once a worker has exited. Every stage has to stay up for the
        whole run, so a dead process means no further result is coming."""
        if not workers:
            return False
        return any(not p.is_alive() for p in workers)

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

    # Claim operations open/close their own connection
    no_more_to_claim = False
    idle_rounds = 0

    # Refill at halfway mark
    claim_batch_size = int(claim_batch_size)
    refill_below = max(1, claim_batch_size // 2)

    while True:
        if not terminating and not no_more_to_claim and inflight < refill_below:
            rows = _claim_batch(cfg, job_id, claim_batch_size, table_name)
            if not rows:
                no_more_to_claim = True
                logging.info(f"No more concatenation jobs to claim for job_id={job_id} at this time.")
            for r in rows:
                task_q.put({k: r[k] for k in r.keys()})
                inflight += 1

        if inflight == 0 and (terminating or no_more_to_claim):
            break

        if collect(1.0):
            idle_rounds = 0
        else:
            idle_rounds += 1
            if idle_rounds >= 3 and workers_gone():
                logging.error(f"A worker exited with {inflight} pair(s) still in flight")
                break

        flush(force=False)

    idle_rounds = 0
    while inflight > 0:
        if collect(5.0):
            idle_rounds = 0
        else:
            idle_rounds += 1
            if idle_rounds >= 3 and workers_gone():
                logging.error(f"A worker exited with {inflight} pair(s) still in flight")
                break
        flush(force=False)

    flush(force=True)
    if terminating:
        raise SystemExit(143)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein concatenation executor')
    parser.add_argument('--config', required=True, help='Project config YAML')
    args = parser.parse_args()

    cfg = load_project_and_system_config(args.config)
    setup_logging(cfg)

    project_path = cfg.get('project').get('project_path')
    if not project_path:
        raise SystemExit('Missing config: project.project_path')
    monomers_path = os.path.join(project_path, cfg.get('paths').get('monomers'))
    output_dir = os.path.join(project_path, cfg.get('paths').get('couplings'))
    os.makedirs(output_dir, mode=0o755, exist_ok=True)

    tables = get_table_names(cfg)
    concat_table = tables['couplings']
    concat_cfg = cfg.get('concatenate')

    down_cfg = concat_cfg.get('downsample') or {}
    downsample = bool(down_cfg.get('enabled', False))
    downsample_to = int(down_cfg.get('to', 30000))
    reweighting_cfg = concat_cfg.get('reweighting') or {}
    reweighting_enabled = bool(reweighting_cfg.get('enabled', True))
    reweighting_threshold = float(reweighting_cfg.get('threshold', 0.8))
    col_gap_threshold = float(cfg.get('align', {}).get('col_gap_threshold', 0.5))

    reweight_threads = 4
    use_gpu = gpu_is_available()
    logical_cpus = int(psutil.cpu_count(logical=True) or os.cpu_count() or 1)
    if use_gpu:
        devices = visible_gpu_devices()
        n_gpu_worker = len(devices)
        # One process per GPU does nothing but drive the device, and every
        # remaining core concatenates, so the GPUs are kept fed.
        n_worker = max(1, logical_cpus - n_gpu_worker)
        ctx = mp.get_context('spawn')
        logging.info(
            f"Concatenating on GPU: {n_worker} concatenation worker(s) feeding "
            f"{n_gpu_worker} reweighting worker(s), one per GPU "
            f"(devices: {', '.join(devices)}) over {logical_cpus} logical cores"
        )
    else:
        devices = []
        n_gpu_worker = 0
        n_worker = max(1, logical_cpus // reweight_threads)
        ctx = mp.get_context()
        logging.info(
            f"Concatenating on CPU: {n_worker} worker(s) x {reweight_threads} "
            f"reweighting thread(s) over {logical_cpus} logical cores"
        )

    claim_batch_size = concat_cfg.get('claim_batch_size', 'auto')
    if claim_batch_size == 'auto':
        claim_batch_size = max(4, n_worker * 32)
    else:
        claim_batch_size = int(claim_batch_size)

    job_id = worker_id()

    task_q = ctx.Queue()
    result_q = ctx.Queue()
    # GPU path only: one request queue into the reweighting pool, and one reply
    # queue per concatenation worker so each gets its own weights back.
    gpu_q = ctx.Queue() if use_gpu else None
    reply_qs = [ctx.Queue() for _ in range(n_worker)] if use_gpu else []
    shutdown_evt = ctx.Event() if use_gpu else None

    concat_opts = dict(
        downsample=downsample,
        downsample_to=downsample_to,
        reweight=reweighting_enabled,
        reweighting_threshold=reweighting_threshold,
        col_gap_threshold=col_gap_threshold,
        n_threads=0 if use_gpu else reweight_threads,
    )

    workers = []
    for i in range(int(n_worker)):
        p = ctx.Process(
            target=_worker_loop,
            kwargs=dict(
                task_q=task_q,
                result_q=result_q,
                cfg=cfg,
                monomers_path=monomers_path,
                output_dir=output_dir,
                concat_opts=concat_opts,
                gpu_q=gpu_q,
                reply_q=reply_qs[i] if use_gpu else None,
                worker_idx=i,
                shutdown_evt=shutdown_evt,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    gpu_workers = []
    for device_id in devices:
        p = ctx.Process(
            target=_gpu_worker_loop,
            kwargs=dict(
                gpu_q=gpu_q,
                reply_qs=reply_qs,
                cfg=cfg,
                device_id=device_id,
            ),
            daemon=True,
        )
        p.start()
        gpu_workers.append(p)

    try:
        coordinator_loop(
            cfg=cfg,
            job_id=job_id,
            task_q=task_q,
            result_q=result_q,
            claim_batch_size=claim_batch_size,
            table_name=concat_table,
            workers=workers + gpu_workers,
            commit_every_tasks=64,
            commit_every_seconds=300,
        )
    finally:
        # Release anyone still waiting on a GPU reply, then stop the
        # concatenation workers before the pool they depend on.
        if shutdown_evt is not None:
            shutdown_evt.set()
        for _ in workers:
            task_q.put(None)
        _drain_and_join(workers, [q for q in (gpu_q, result_q) if q is not None])
        for _ in gpu_workers:
            gpu_q.put(None)
        _drain_and_join(gpu_workers, reply_qs)

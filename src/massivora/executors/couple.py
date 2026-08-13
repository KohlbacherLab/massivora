import argparse
import logging
import multiprocessing as mp
import os
import signal
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import shared_memory

import dask
import numpy as np
import psutil
import zarr
from dask import annotate, delayed
from dask.distributed import (Client, Nanny, Scheduler, SpecCluster,
                              TimeoutError, WorkerPlugin, wait)

from massivora import cpp_bindings
from massivora.alignment import BinaryAlignment
from massivora.config import load_project_and_system_config
from massivora.db import (STATUS, compute_file_hash, connect_db_rw,
                          ensure_columns, get_db_path, get_hashed_file_path,
                          get_table_names, quote_identifier)
from massivora.utils import (SHM_PREFIX, compute_id_range, get_cuda_module,
                             gpu_is_available, massivora_pkg_dir,
                             setup_logging, worker_id)
if gpu_is_available():
    import cupy as cp


class LoggingWorkerPlugin(WorkerPlugin):
    def __init__(self, cfg):
        self._cfg = cfg

    def setup(self, worker):
        setup_logging(self._cfg)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseCouplingExecutor(object):
    SHM_PREFIX = SHM_PREFIX
    def __init__(self, config):
        self._config = config

        project_path = config['project']['project_path']
        if not project_path:
            raise ValueError('Missing config: project.project_path')
        self.project_path = project_path

        self.db_path = get_db_path(config)
        self.alignment_dir = os.path.join(project_path, config['paths']['monomers'])
        self.output_dir = os.path.join(project_path, config['paths']['couplings'])

        down_cfg = config.get('concatenate').get('downsample')
        self.downsample = bool(down_cfg.get('enabled', False))
        self.downsample_to = int(down_cfg.get('to', 30000))
        self.reweighting_threshold = float(
            config.get('concatenate').get('reweighting_threshold', 0.8))

        coupling_cfg = config.get('couple')
        tables = get_table_names(config)
        self.align_table = tables['alignments']
        self.couple_table = tables['couplings']
        self.lambda_h = float(coupling_cfg.get('lambda_h', 0.01))
        self.lambda_J = float(coupling_cfg.get('lambda_J', 0.01))
        self.eps_conv = float(coupling_cfg.get('eps_conv', 3e-4))
        self.maxit = int(coupling_cfg.get('maxit', 200))

        claim = coupling_cfg.get('claim_batch_size', 'auto')
        if claim == 'auto':
            self.claim_batch_size = max(4, int(mp.cpu_count() / 4))
        else:
            self.claim_batch_size = int(claim)

        self.portion = float(coupling_cfg.get('portion', 1.0))
        self.portion_start = float(coupling_cfg.get('portion_start', 0.0))

        self.job_id = worker_id()
        self.plm_opt_exe = None

        ensure_columns(self._config)

        # Maintain database connection for the lifetime of this executor
        self.db_conn = connect_db_rw(self._config)

        # Set row factory for dict-like access (SQLite-specific)
        if hasattr(self.db_conn, 'row_factory'):
            self.db_conn.row_factory = sqlite3.Row

    def close_db_connection(self):
        """Close database connection."""
        if hasattr(self, 'db_conn') and self.db_conn:
            try:
                self.db_conn.close()
            except Exception as e:
                logging.error(f"Failed to close DB connection for worker {self.job_id}: {e}")

    def __getstate__(self):
        """Exclude db_conn from pickling (workers don't need DB connection)."""
        state = self.__dict__.copy()
        state['db_conn'] = None
        return state

    def __setstate__(self, state):
        """Restore state without DB connection."""
        self.__dict__.update(state)

    def concatenate(self, pair, alignment_dir, downsample=False,
                    downsample_to=None):
        pair_name = pair.get('name')
        pid1 = pair.get('pid1')
        pid2 = pair.get('pid2')
        if os.path.exists(os.path.join(alignment_dir, pair_name)):
            return BinaryAlignment(os.path.join(alignment_dir, pair_name))
        align1 = BinaryAlignment(os.path.join(alignment_dir, pid1))
        align2 = BinaryAlignment(os.path.join(alignment_dir, pid2))
        concatenated = align1 + align2
        if downsample:
            concatenated.Downsample_Randomly(to=downsample_to)
        concatenated.Gap_Columns_Control(self._config.get('align').get('col_gap_threshold', 0.5))
        return concatenated

    # -- DB helpers --

    def _claim_batch(self, job_id, batch_size, id_range=None):
        n = int(batch_size)
        if n <= 0:
            return []

        cursor = self.db_conn.cursor()

        start_id = None
        end_id = None
        if id_range is not None:
            try:
                start_id, end_id = id_range
            except Exception:
                logging.error(f"Invalid id_range format: {id_range}, expected (start_id, end_id)")

        table = quote_identifier(self.couple_table)

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

            # For couple job, status null is unclaimable
            cursor.execute(
                f"""
                WITH to_claim AS (
                    SELECT id, status AS prev_status
                    FROM {table}
                    WHERE status IN (:noopt, :pending)
                    AND claimed_at IS NULL
                    {where_range_sql}
                    ORDER BY id
                    LIMIT :n
                )
                UPDATE {table}
                SET status = CASE
                        WHEN id IN (SELECT id FROM to_claim WHERE prev_status = :noopt) THEN :running
                        ELSE status + 1
                    END,
                    job_id = :job_id,
                    claimed_at = :now
                WHERE id IN (SELECT id FROM to_claim)
                RETURNING *
                """,
                params,
            )
            rows = cursor.fetchall()
            self.db_conn.commit()
            logging.debug(f"job_id={job_id} claimed {len(rows)} coupling jobs in {time.time() - start:.3f}s")
            return rows
        except Exception as e:
            logging.error(f"Failed to claim batch of coupling jobs for job_id={job_id}: {e}. Rolling back DB transaction.")
            self.db_conn.rollback()
            raise

    def _db_apply_results(self, done_updates, fail_updates):
        if not done_updates and not fail_updates:
            return 0

        cursor = self.db_conn.cursor()
        table = quote_identifier(self.couple_table)

        try:
            n = 0
            if done_updates:
                cursor.executemany(
                    f"""
                    UPDATE {table}
                    SET status = ?,
                        effnumber = ?,
                        number = ?,
                        length = ?,
                        file_hash = ?,
                        claimed_at = NULL,
                        job_id = NULL
                    WHERE id = ?
                    """,
                    [
                        (
                            x.get('status', STATUS['DONE']),
                            x.get('effnumber'),
                            int(x.get('number')),
                            int(x.get('length')),
                            x.get('file_hash'),
                            x.get('id'),
                        )
                        for x in done_updates
                    ],
                )
                n += len(done_updates)

            if fail_updates:
                cursor.executemany(
                    f"""
                    UPDATE {table}
                    SET status = MIN(status + 1, ?),
                        claimed_at = NULL,
                        job_id = NULL
                    WHERE id = ?
                    """,
                    [(STATUS['FAILED'], i) for i in fail_updates],
                )
                n += len(fail_updates)

            self.db_conn.commit()
            return n
        except Exception as e:
            logging.error(f"Failed to apply results to DB for job_id={self.job_id}: {e}")
            self.db_conn.rollback()
            raise

    def _checkpoint_wal(self):
        """Merge the SQLite WAL into the main database file.

        No-op for non-SQLite backends (MySQL/PostgreSQL) and harmless for
        SQLite connections not using WAL (e.g. DELETE journal mode on network
        filesystems), where ``wal_checkpoint`` returns without effect.
        """
        if not isinstance(self.db_conn, sqlite3.Connection):
            return
        try:
            row = self.db_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            # row = (busy, log_frames, checkpointed_frames); busy != 0 means a
            # concurrent reader/writer prevented a full checkpoint this time.
            if row is not None and row[0] != 0:
                logging.debug(
                    f"WAL checkpoint for job_id={self.job_id} could not fully "
                    f"complete (busy); {row[2]}/{row[1]} frames checkpointed"
                )
        except sqlite3.Error as e:
            logging.warning(f"WAL checkpoint failed for job_id={self.job_id}: {e}")

    def _check_shared_memory(self, inflight):
        """Unlink shared-memory segments this coordinator has leaked.

        Enumeration relies on POSIX shared memory being exposed under
        ``/dev/shm`` (Linux). On platforms without it (macOS/Windows) this is a
        no-op.
        """
        shm_dir = "/dev/shm"
        if not os.path.isdir(shm_dir):
            return

        try:
            existing = [n for n in os.listdir(shm_dir) if n.startswith(self.SHM_PREFIX)]
        except OSError as e:
            logging.warning(f"Could not list {shm_dir} for shared-memory cleanup: {e}")
            return
        if not existing:
            return

        # Map each segment back to a candidate pair name: strip our prefix, and
        # for the coupling buffer also strip the trailing "_J".
        prefix_len = len(self.SHM_PREFIX)
        candidates = set()
        for seg in existing:
            core = seg[prefix_len:]
            candidates.add(core)
            if core.endswith("_J"):
                candidates.add(core[:-2])

        # Resolve ownership (pair id + owning job_id) for every candidate in one query.
        table = quote_identifier(self.couple_table)
        placeholders = ",".join("?" * len(candidates))
        try:
            cur = self.db_conn.execute(
                f"SELECT id, name, job_id FROM {table} WHERE name IN ({placeholders})",
                tuple(candidates),
            )
            owners = {row[1]: (row[0], row[2]) for row in cur.fetchall()}
        except Exception as e:
            logging.warning(f"Shared-memory cleanup skipped; DB lookup failed: {e}")
            return

        inflight_ids = set(inflight.values())
        me = str(self.job_id)

        removed = 0
        for seg in existing:
            core = seg[prefix_len:]
            # Prefer an exact name match (the MSA buffer, or a pair literally
            # named like the core); otherwise treat it as the "_J" buffer.
            if core in owners:
                pair_name = core
            elif core.endswith("_J") and core[:-2] in owners:
                pair_name = core[:-2]
            else:
                # No known pair owns this segment name; leave it alone rather
                # than reason about something we can't attribute.
                continue

            pair_id, owner = owners[pair_name]
            if pair_id in inflight_ids:
                continue  # our own active pair
            if owner is not None and str(owner) != me:
                continue  # claimed by another live coordinator on this node

            try:
                shm = shared_memory.SharedMemory(name=seg)
                shm.close()
                shm.unlink()
                removed += 1
            except FileNotFoundError:
                # Already unlinked by the owning task in the meantime.
                pass
            except Exception as e:
                logging.warning(f"Failed to unlink stale shared memory '{seg}': {e}")

        if removed:
            logging.info(
                f"Cleaned up {removed} stale shared-memory segment(s) for job_id={self.job_id}"
            )

    @staticmethod
    def compute_score(J, q=21, gap_idx=0):
        """
        Compute the coupling scores with Frobenius norm and APC

        Parameters
        ----------
        `J` — np.ndarray
            The input J matrix. It has to be the shape (N*q*q, N)
        `q` — int (optional)
            Number of simbols in the J matrix (default: `21`)
        `gap_idx` — int (optional)
            Index of gap in the encoding (default: `0`)

        Returns
        -------
        `np.ndarray`
            The pairwise coupling matrix
        """
        # Average the J matrix out at main diagonal
        N = J.shape[1]
        J = J.T.reshape((N, N, q, q))
        J = (J + J.transpose(1, 0, 3, 2)) / 2

        # Gap exclude Frobenius norm
        mask = np.ones((q, q), dtype=bool)
        mask[gap_idx, :] = False
        mask[:, gap_idx] = False
        J_filtered = J[:, :, mask].reshape(N, N, -1)
        FN = np.sqrt(np.sum(J_filtered ** 2, axis=-1))

        # Average Product Correction of Frobenius norm
        row_mean = np.sum(FN, axis=1, keepdims=True) / (N - 1)
        col_mean = np.sum(FN, axis=0, keepdims=True) / (N - 1)
        total_mean = np.sum(FN) / (N * (N - 1))
        FN_APC = FN - (row_mean @ col_mean) / total_mean

        return FN_APC

    # -- Abstract methods --

    def _create_cluster(self):
        """Return (cluster, client). Subclass must override."""
        raise NotImplementedError("Subclass must implement _create_cluster()")

    def calc_graph(self, pair, priority, **kwargs):
        """Build a Dask delayed graph for one pair. Subclass must override."""
        raise NotImplementedError("Subclass must implement calc_graph()")

    # -- Entry point --

    def run(self, client=None):
        """Main entry point. Creates a cluster if no client is provided."""
        cursor = self.db_conn.cursor()
        cursor.execute(f"SELECT COALESCE(MAX(id), 0) FROM {quote_identifier(self.couple_table)}")
        total = int(cursor.fetchone()[0] or 0)

        id_range = compute_id_range(self.portion, self.portion_start, total)

        owns_client = client is None
        cluster = None
        if owns_client:
            cluster, client = self._create_cluster()

        try:
            client.forward_logging()
            client.register_plugin(
                LoggingWorkerPlugin(self._config),
                name="massivora-logging",
            )
            self.coordinator_loop(dask_client=client, id_range=id_range)
        finally:
            if owns_client:
                client.close()
                if cluster is not None:
                    cluster.close()
            self.close_db_connection()

    # -- Coordinator loop --

    def coordinator_loop(self,
        dask_client=None,
        id_range=None,
        commit_every_tasks=64,
        commit_every_seconds=300,
    ):
        """Single-process coordinator:

        - Claims work from DB and feeds it to workers.
        - Receives results and performs DB updates.
        - Commits (and checkpoints the WAL into the main DB file) every 64
          tasks OR every 5 minutes.

        Lease-expiry reclaim is intentionally NOT implemented here.

        Termination behavior:
        - SIGTERM/SIGINT: flush DB immediately, then stop claiming new work, drain results briefly,
        flush again, and exit.
        - SIGKILL cannot be handled in user-space.
        """
        inflight = {}

        done_updates = []
        fail_ids = []

        last_commit = float(time.time())
        terminating = False

        def flush(force=False):
            nonlocal done_updates, fail_ids, last_commit
            if not done_updates and not fail_ids:
                return 0

            if not force:
                if (len(done_updates) + len(fail_ids)) < int(commit_every_tasks) and (time.time() - last_commit) < float(commit_every_seconds):
                    return 0

            n = self._db_apply_results(done_updates, fail_ids)
            self._checkpoint_wal()
            self._check_shared_memory(inflight)
            done_updates = []
            fail_ids = []
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

        # Only coordinator handles signals; workers just compute.
        signal.signal(signal.SIGTERM, _on_term)
        signal.signal(signal.SIGINT, _on_term)

        # First-in, first-out counter
        submit_counter = 0

        def _submit_one(task_row):
            nonlocal submit_counter
            pair = task_row

            graph = self.calc_graph(
                pair=pair,
                priority=submit_counter,
                alignment_dir=self.alignment_dir,
                output_dir=self.output_dir,
                downsample=self.downsample,
                downsample_to=self.downsample_to,
                reweighting_threshold=self.reweighting_threshold,
                lambdaH=self.lambda_h,
                lambdaJ=self.lambda_J,
                eps_conv=self.eps_conv,
                maxit=self.maxit,
                plm_opt_exe=self.plm_opt_exe
            )
            submit_counter -= 1

            return dask_client.compute(graph)

        no_more_to_claim = False

        # seed / main loop
        while True:
            if not terminating and not no_more_to_claim:
                while len(inflight) < (self.claim_batch_size/2):
                    rows = self._claim_batch(self.job_id, self.claim_batch_size, id_range=id_range)
                    if not rows:
                        no_more_to_claim = True
                        logging.info(f"No more coupling jobs to claim for job_id={self.job_id} at this time.")
                        break

                    for r in rows:
                        task_row = {k: r[k] for k in r.keys()}
                        future = _submit_one(task_row)
                        inflight[future] = task_row['id']

            if not inflight or terminating:
                break

            try:
                done, pending = wait(list(inflight.keys()), timeout=1.0, return_when='FIRST_COMPLETED')
            except TimeoutError:
                pass
            else:
                for future in done:
                    task_id = inflight.pop(future, None)
                    if task_id is None:
                        continue
                    try:
                        res = future.result()
                    except Exception:
                        logging.exception(f"Task {task_id} failed")
                        fail_ids.append(task_id)
                    else:
                        done_updates.append(res)
                        future.release()

            # periodic flush
            flush(force=False)

        # final flush
        flush(force=True)

        if terminating:
            self.close_db_connection()
            raise SystemExit(143)


# ---------------------------------------------------------------------------
# CPU PLM executor
# ---------------------------------------------------------------------------

class PLMCouplingExecutor(BaseCouplingExecutor):
    def __init__(self, config):
        super().__init__(config)
        massivora_dir = massivora_pkg_dir()
        self.plm_opt_exe = os.path.join(massivora_dir, 'bin', 'plm_opt_site')

    def _create_cluster(self):
        total_cpus = mp.cpu_count()
        loader_threads = min(total_cpus // 5, 8)
        compute_threads = max(1, total_cpus - loader_threads - 1)
        loader_compute_slots = max(1, loader_threads - 1)

        logging.info(
            f"Creating cluster: loader=1 worker ({loader_threads} threads, "
            f"{loader_compute_slots} compute slots), "
            f"compute=1 worker ({compute_threads} threads)"
        )

        worker_spec = {
            'loader': {
                'cls': Nanny,
                'options': {
                    'nthreads': loader_threads,
                    'resources': {
                        'loader': loader_threads,
                        'compute': loader_compute_slots,
                    },
                    'memory_limit': 0,
                    'host': '127.0.0.1',
                    'death_timeout': 120,
                },
            },
            'compute': {
                'cls': Nanny,
                'options': {
                    'nthreads': compute_threads,
                    'resources': {'compute': compute_threads},
                    'memory_limit': 0,
                    'host': '127.0.0.1',
                    'death_timeout': 120,
                },
            },
        }

        cluster = SpecCluster(
            workers=worker_spec,
            scheduler={
                'cls': Scheduler,
                'options': {
                    'host': '127.0.0.1',
                    'dashboard_address': None,
                }
            }
        )
        client = Client(cluster)
        client.wait_for_workers(2, timeout=60)
        return cluster, client

    def load_pair(self, pair, **kwargs):
        pair_name = pair.get('name')
        pair_id = pair.get('id')
        pid1 = pair.get('pid1')
        pid2 = pair.get('pid2')
        alignment_dir = kwargs.get('alignment_dir')
        output_dir = kwargs.get('output_dir')
        downsample = kwargs.get('downsample', False)
        downsample_to = kwargs.get('downsample_to', None)
        reweighting_threshold = kwargs.get('reweighting_threshold', 0.8)
        lambdaH = kwargs.get('lambdaH', 0.01)
        lambdaJ = kwargs.get('lambdaJ', 0.01)
        eps_conv = kwargs.get("eps_conv", 1e-4)
        maxit = kwargs.get("maxit", 500)
        plm_opt_exe = kwargs.get("plm_opt_exe")

        # Compute file hash for this coupling
        file_hash = pair.get('file_hash') or compute_file_hash(pair_name)
        filename, _ = get_hashed_file_path(pair_name, file_hash, output_dir)

        logging.debug(f"Loading pair id {pair_id}: {pair.get('pid1')}-{pair.get('pid2')}")

        # Try to load pre-concatenated alignment; if it fails, concatenate on-the-fly and save for future reuse.
        try:
            align = BinaryAlignment(filename)
        except Exception as e:
            logging.warning(f"Failed to load pair {pair_id} '{pair_name}', trying to concatenate.")
            align = self.concatenate(pair, alignment_dir, downsample=downsample,
                                    downsample_to=downsample_to)
            align.Reweight_Sequence(reweighting_threshold)
            align.To_Zarr(filename, overwrite=True)

        MSA = align.matrix.astype(np.int32)
        B, N = MSA.shape
        W = np.array(align.weights, dtype=np.float64)
        Beff = float(align.Beff)
        W /= Beff

        q = int(MSA.max()) + 1

        # Create MSA shared memory
        msa_size = MSA.nbytes
        w_size = W.nbytes
        logging.debug(f"MSA size (bytes): {msa_size}, Weights size (bytes): {w_size}")
        logging.debug(f"MSA dimensions: {MSA.shape}, Weights length: {W.shape}")
        msa_shm = shared_memory.SharedMemory(name=self.SHM_PREFIX +pair_name, create=True, size=msa_size+w_size)
        msa_arr = np.ndarray(MSA.shape, dtype=np.int32, buffer=msa_shm.buf)
        w_arr = np.ndarray(W.shape, dtype=np.float64, buffer=msa_shm.buf[msa_size:])
        np.copyto(msa_arr, MSA)
        np.copyto(w_arr, W)

        shared = {
            "name": pair_name,
            "B": B,
            "N": N,
            "q": q,
            "Beff": Beff,
            "lambdaH": float(lambdaH),
            "lambdaJ": float(lambdaJ),
            "pid1": pid1,
            "pid2": pid2,
            "pair_id": pair_id,
            "alignment_dir": alignment_dir,
            "output_dir": output_dir,
            "eps_conv": eps_conv,
            "maxit": maxit,
            "plm_opt_exe": plm_opt_exe,
            "file_hash": file_hash
        }
        return shared

    def optimize_site(self, r, params):
        q = params["q"]
        N = params["N"]
        B = params["B"]
        msa_name = params["name"]
        lambdaH, lambdaJ = params["lambdaH"], params["lambdaJ"]
        eps_conv = params["eps_conv"]
        maxit = params["maxit"]
        plm_opt_exe = params["plm_opt_exe"]

        if r >= N: 
            logging.error(f"Invalid site index r={r} for {msa_name}, whose N={N}")
            return r, 1  # Invalid site index; return non-zero code to indicate failure

        try:
            J_shm = shared_memory.SharedMemory(name=self.SHM_PREFIX +msa_name+'_J', create=True, size=N*N*q*q*8)
        except FileExistsError:
            pass # Ensure the shared memory has been created, do nothing here
        logging.debug(F"Running PLM optimization with cmd {' '.join([plm_opt_exe, msa_name, str(r), str(B), str(N), str(q), str(lambdaH), str(lambdaJ), str(eps_conv), str(maxit)])}")
        rv = subprocess.run([plm_opt_exe, msa_name, str(r), str(B), str(N), str(q), str(lambdaH), str(lambdaJ), str(eps_conv), str(maxit)])

        return r, rv.returncode

    def collect_results(self, results, params):
        failed_sites = []
        for item in results:
            r, returncode = item
            if returncode != 0:
                failed_sites.append((r, returncode))

        if failed_sites:
            details = ", ".join(f"site {r}: rc={rc}" for r, rc in failed_sites)
            logging.error(
                f"Optimization failed for pair {params['pair_id']} '{params['name']}' at {details}"
            )
        output_dir = params["output_dir"]
        msa_name = params["name"]
        file_hash = params.get("file_hash")
        filename, _ = get_hashed_file_path(msa_name, file_hash, output_dir)
        N = params["N"]
        q = params["q"]

        J_shm = shared_memory.SharedMemory(name=self.SHM_PREFIX +params["name"]+'_J', size=N*N*q*q*8)
        J = np.ndarray((N*q*q, N), dtype=np.float64, buffer=J_shm.buf)

        score = self.compute_score(J, q=q).astype(np.float16)
        logging.info(f"Writing coupling score for pair {params['pair_id']} '{msa_name}' to alignment zarr")
        grp = zarr.open_group(store=filename)
        # TODO: Compression need implementing
        z = grp.require_array(name=self.couple_table, shape=score.shape, dtype='float16', overwrite=True)
        z[:] = score

        result = {
            'id': params["pair_id"],
            'effnumber': params["Beff"],
            'length': int(params["N"]),
            'number': int(params["B"]),
            'file_hash': params.get("file_hash"),
        }
        try:
            del J
            J_shm.close()
            J_shm.unlink()
            shm = shared_memory.SharedMemory(name=self.SHM_PREFIX +params["name"])
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        return result

    def calc_graph(self, pair, priority=0, **kwargs):
        N = pair.get('length')
        if not N:
            length1 = self.db_conn.execute(f"select length from {quote_identifier(self.align_table)} where pid = ?", (pair.get('pid1'),)).fetchone()[0]
            length2 = self.db_conn.execute(f"select length from {quote_identifier(self.align_table)} where pid = ?", (pair.get('pid2'),)).fetchone()[0]
            N = length1 + length2

        with annotate(resources={'loader': 1}, priority=0):
            params = delayed(self.load_pair)(pair, **kwargs)

        with annotate(resources={'compute': 1}, priority=priority):
            site_tasks = [delayed(self.optimize_site)(r, params) for r in range(int(N))]
            out_d = delayed(self.collect_results)(site_tasks, params)

        return out_d


# ---------------------------------------------------------------------------
# GPU PLM executor
# ---------------------------------------------------------------------------

class PLMCouplingExecutorGPU(BaseCouplingExecutor):
    def __init__(self, config):
        super().__init__(config)
        massivora_dir = massivora_pkg_dir()
        # Same executable as the CPU path; invoked with --use-gpu so it runs the
        # CUDA code (requires the binary to be built with -DENABLE_CUDA=ON).
        self.plm_opt_exe = os.path.join(massivora_dir, 'bin', 'plm_opt_site')

    def _create_cluster(self):
        """Create a Dask cluster with one worker per GPU."""
        devices = None
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_visible is not None and cuda_visible.strip() != "":
            devices = [d.strip() for d in cuda_visible.split(',') if d.strip()]
            if not devices:
                logging.warning("CUDA_VISIBLE_DEVICES is set but empty; falling back to all GPUs")
                devices = None

        if devices is None:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                    capture_output=True, text=True, check=True)
                devices = [d.strip() for d in result.stdout.strip().split('\n') if d.strip()]
            except Exception:
                devices = ['0']
                logging.warning("Could not detect GPU count; defaulting to device 0")

        n_gpus = len(devices)
        compute_threads = max(1, n_gpus * 2)
        logging.info(f"Creating GPU cluster with {n_gpus} workers (one per GPU)")

        worker_spec = {
            'cpu': {
                'cls': Nanny,
                'options': {
                    'nthreads': compute_threads,
                    'resources': {'cpu': compute_threads},
                    'memory_limit': 0,
                    'host': '127.0.0.1',
                    'death_timeout': 120,
                },
            }
        }
        for i, device_id in enumerate(devices):
            worker_spec[f'gpu-{i}'] = {
                'cls': Nanny,
                'options': {
                    'nthreads': 1,
                    'resources': {'gpu': 1},
                    'memory_limit': 0,
                    'host': '127.0.0.1',
                    'death_timeout': 120,
                    'env': {'CUDA_VISIBLE_DEVICES': str(device_id)},
                },
            }

        cluster = SpecCluster(
            workers=worker_spec,
            scheduler={
                'cls': Scheduler,
                'options': {
                    'host': '127.0.0.1',
                    'dashboard_address': None,
                }
            }
        )
        client = Client(cluster)
        client.wait_for_workers(n_gpus, timeout=120)
        return cluster, client

    def load_pair(self, pair, **kwargs):
        pair_name = pair.get('name')
        pair_id = pair.get('id')
        alignment_dir = kwargs.get('alignment_dir')
        output_dir = kwargs.get('output_dir')
        downsample = kwargs.get('downsample', False)
        downsample_to = kwargs.get('downsample_to', None)
        reweighting_threshold = kwargs.get('reweighting_threshold', 0.8)

        # Compute file hash for this coupling
        file_hash = pair.get('file_hash') or compute_file_hash(pair_name)
        filename, _ = get_hashed_file_path(pair_name, file_hash, output_dir)

        logging.debug(f"Loading pair id {pair_id}: {pair.get('pid1')}-{pair.get('pid2')}")

        # Try to load pre-concatenated alignment; if it fails, concatenate on-the-fly and save for future reuse.
        try:
            align = BinaryAlignment(filename)
        except Exception as e:
            logging.warning(f"Failed to load pair {pair_id} '{pair_name}', trying to concatenate.")
            align = self.concatenate(pair, alignment_dir, downsample=downsample,
                                    downsample_to=downsample_to)
            align.Reweight_Sequence(reweighting_threshold, use_GPU=True)
            align.To_Zarr(filename, overwrite=True)

        MSA = np.asarray(align.matrix, dtype=np.int32)
        B, N = MSA.shape
        W = np.asarray(align.weights, dtype=np.float64)
        Beff = float(align.Beff)
        W /= Beff
        q = int(MSA.max()) + 1

        # Create MSA shared memory
        msa_size = MSA.nbytes
        w_size = W.nbytes
        logging.debug(f"Pair {pair_id} '{pair_name}': MSA size (bytes): {msa_size}, Weights size (bytes): {w_size}")
        logging.debug(f"Pair {pair_id} '{pair_name}': MSA dimensions: {MSA.shape}, Weights length: {W.shape}")
        try:
            msa_shm = shared_memory.SharedMemory(name=self.SHM_PREFIX +pair_name, create=True, size=msa_size+w_size)
        except FileExistsError:
            logging.warning(f"Shared memory for pair {pair_id} '{pair_name}' already exists; reusing it.")
            msa_shm = shared_memory.SharedMemory(name=self.SHM_PREFIX +pair_name)
        msa_arr = np.ndarray(MSA.shape, dtype=np.int32, buffer=msa_shm.buf)
        w_arr = np.ndarray(W.shape, dtype=np.float64, buffer=msa_shm.buf[msa_size:])
        np.copyto(msa_arr, MSA)
        np.copyto(w_arr, W)

        # Update kwargs with computed values and ensure all needed fields are present
        kwargs.update({
            "name": pair_name,
            "pair_id": pair_id,
            "B": B,
            "N": N,
            "q": q,
            "MSA": MSA,
            "W": W,
            "Beff": Beff,
            "file_hash": file_hash
        })
        del align

        return kwargs

    def optimize_pair(self, pair, params):
        msa_name = params["name"]
        q = params["q"]
        N = params["N"]
        B = params["B"]
        lambdaH, lambdaJ = params["lambdaH"], params["lambdaJ"]
        eps_conv = params["eps_conv"]
        maxit = params["maxit"]
        n_streams = int(params.get("n_streams", 4))
        plm_opt_exe = params["plm_opt_exe"]

        try:
            shared_memory.SharedMemory(name=self.SHM_PREFIX +msa_name+'_J', create=True, size=N*N*q*q*4)
        except FileExistsError:
            pass  # Ensure the shared memory has been created, do nothing here

        cmd = [plm_opt_exe, "--use-gpu", msa_name, str(B), str(N), str(q),
               str(lambdaH), str(lambdaJ), str(eps_conv), str(maxit), str(n_streams)]
        logging.debug(f"Running GPU PLM optimization with cmd {' '.join(cmd)}")
        rv = subprocess.run(cmd)

        return rv.returncode

    def collect_results(self, results, params):
        N = params["N"]
        output_dir = params["output_dir"]
        msa_name = params["name"]
        file_hash = params.get("file_hash")
        filename, _ = get_hashed_file_path(msa_name, file_hash, output_dir)
        q = params["q"]

        if results != 0:
            try:
                J_shm = shared_memory.SharedMemory(name=self.SHM_PREFIX +params["name"]+'_J', size=N*N*q*q*4)
                J_shm.close()
                J_shm.unlink()
                shm = shared_memory.SharedMemory(name=self.SHM_PREFIX +params["name"])
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            raise RuntimeError(
                f"GPU optimization failed for pair {params['pair_id']} (rc={results})"
            )

        # The CUDA executable stores J as float32 with one row per site:
        # row r holds site r's (N, q, q) coupling blocks flattened to (N*q*q,).
        J_shm = shared_memory.SharedMemory(name=self.SHM_PREFIX +params["name"]+'_J', size=N*N*q*q*4)
        J = np.ndarray((N, N*q*q), dtype=np.float32, buffer=J_shm.buf)

        cpp_bindings.applyIsingGauge(J, q)
        score = self.compute_score(J.T, q=q).astype(np.float16)

        logging.info(f"Writing coupling score for pair {params['pair_id']} '{msa_name}' to alignment zarr")
        grp = zarr.open_group(store=filename)
        # TODO: Compression need implementing
        z = grp.require_array(name=self.couple_table, shape=score.shape, dtype='float16', overwrite=True)
        z[:] = score

        # Release GPU buffers and free CuPy pools to return memory to the driver.
        del z, grp, score

        result = {
            'id': params["pair_id"],
            'effnumber': params["Beff"],
            'length': int(params["N"]),
            'number': int(params["B"]),
            'file_hash': params.get("file_hash"),
        }
        try:
            del J
            J_shm.close()
            J_shm.unlink()
            shm = shared_memory.SharedMemory(name=self.SHM_PREFIX +params["name"])
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        return result

    def calc_graph(self, pair, priority, **kwargs):
        with annotate(resources={'cpu': 1}, priority=0):
            params = delayed(self.load_pair)(pair, **kwargs)
        with annotate(resources={'gpu': 1}, priority=priority):
            results = delayed(self.optimize_pair)(pair, params)
        with annotate(resources={'cpu': 1}, priority=priority):
            out_d = delayed(self.collect_results)(results, params)
        return out_d


# ---------------------------------------------------------------------------
# GaussDCA executor
# ---------------------------------------------------------------------------

class GaussCouplingExecutor(BaseCouplingExecutor):
    def __init__(self, config):
        super().__init__(config)
        coupling_cfg = config.get('couple')
        self.pseudocount = float(coupling_cfg.get('pseudocount', 0.8))
        self.n_threads = max(1, int(coupling_cfg.get('gauss_threads', 1)))
        massivora_dir = massivora_pkg_dir()
        self.gauss_infer_exe = os.path.join(massivora_dir, 'bin', 'gauss_infer')

        # Memory-aware admission control.
        self.mem_limit = float(coupling_cfg.get('memory_limit', 0.9))
        self.mem_per_pair_factor = float(coupling_cfg.get('mem_per_pair_factor', 1.10))
        self.cov_bytes = self._probe_cov_bytes(coupling_cfg)
        self.mem_budget_mb = max(1.0, self.mem_limit * psutil.virtual_memory().total / 1e6)

    def _probe_cov_bytes(self, coupling_cfg):
        """Width of the covariance scalar the installed gauss_infer uses."""
        configured = coupling_cfg.get('cov_bytes')
        try:
            out = subprocess.run([self.gauss_infer_exe, '--info'],
                                 capture_output=True, text=True, timeout=30)
            info = dict(l.split('=', 1) for l in out.stdout.splitlines() if '=' in l)
            if info.get('lapack') == '0':
                logging.warning(
                    "%s was built without a LAPACK; the covariance inverse will run "
                    "single-threaded through the Eigen fallback (roughly 50x slower "
                    "on wide alignments). Rebuild with openblas available.",
                    self.gauss_infer_exe)
            if configured:
                return int(configured)
            if 'cov_bytes' in info:
                return int(info['cov_bytes'])
        except Exception as e:
            logging.warning(f"Could not probe {self.gauss_infer_exe} for its precision "
                            f"({e}); assuming float64. A single-precision build will "
                            f"then be over-reserved by 2x.")
        return 8

    def _pair_memory_mb(self, N):
        """Peak resident memory of one gauss_infer run, in MB."""
        NQ = int(N) * 20
        pair_bytes = (self.cov_bytes * NQ * NQ // 2   # covariance, lower triangle only
                      + 4 * int(N) * int(N)           # score segment
                      + 10_000 * NQ                   # LAPACK/BLAS workspace, grows with NQ
                      + 96 * 1024 * 1024)             # runtime, libraries, alignment buffers
        return self.mem_per_pair_factor * pair_bytes / 1e6

    def _create_cluster(self):
        total_cpus = mp.cpu_count()
        loader_threads = min(total_cpus // 5, 8)
        compute_threads = max(1, total_cpus - loader_threads - 1)
        loader_compute_slots = max(1, loader_threads - 1)

        logging.info(
            f"Creating GaussDCA cluster: loader=1 worker ({loader_threads} threads, "
            f"{loader_compute_slots} compute slots), "
            f"compute=1 worker ({compute_threads} threads), "
            f"memory budget {self.mem_budget_mb:.0f} MB"
        )

        worker_spec = {
            'loader': {
                'cls': Nanny,
                'options': {
                    'nthreads': loader_threads,
                    'resources': {
                        'loader': loader_threads,
                        'compute': loader_compute_slots,
                    },
                    'memory_limit': 'auto',
                    'host': '127.0.0.1',
                    'death_timeout': 120,
                },
            },
            'compute': {
                'cls': Nanny,
                'options': {
                    'nthreads': compute_threads,
                    'resources': {
                        'compute': compute_threads,
                        'memory_mb': self.mem_budget_mb,
                    },
                    'memory_limit': 'auto',
                    'host': '127.0.0.1',
                    'death_timeout': 120,
                },
            },
        }

        cluster = SpecCluster(
            workers=worker_spec,
            scheduler={
                'cls': Scheduler,
                'options': {
                    'host': '127.0.0.1',
                    'dashboard_address': None,
                }
            }
        )
        client = Client(cluster)
        client.wait_for_workers(2, timeout=60)
        return cluster, client

    @staticmethod
    def compute_score(J, q=21):
        Q = q - 1
        NQ = J.shape[0]
        if NQ % Q != 0:
            raise ValueError(f"J size {NQ} is not a multiple of q-1={Q}")
        N = NQ // Q

        # Block view Jb[i, a, j, b] = J[i*Q + a, j*Q + b]; block (i, j) is e_kl.
        Jb = J.reshape(N, Q, N, Q)

        # Ising gauge 
        row_mean = Jb.mean(axis=3, keepdims=True)
        col_mean = Jb.mean(axis=1, keepdims=True)
        grand = Jb.mean(axis=(1, 3), keepdims=True)
        e_zsg = Jb - row_mean - col_mean + grand
        FN = np.sqrt(np.sum(e_zsg ** 2, axis=(1, 3)))
        np.fill_diagonal(FN, 0.0)

        # Average Product Correction
        Si = FN.sum(axis=0, keepdims=True)
        Sj = FN.sum(axis=1, keepdims=True)
        Sa = FN.sum() * (1.0 - 1.0 / N)
        return FN - (Sj @ Si) / Sa

    def load_pair(self, pair, **kwargs):
        pair_name = pair.get('name')
        pair_id = pair.get('id')
        alignment_dir = kwargs.get('alignment_dir')
        output_dir = kwargs.get('output_dir')
        downsample = kwargs.get('downsample', False)
        downsample_to = kwargs.get('downsample_to', None)
        reweighting_threshold = kwargs.get('reweighting_threshold', 0.8)

        file_hash = pair.get('file_hash') or compute_file_hash(pair_name)
        filename, _ = get_hashed_file_path(pair_name, file_hash, output_dir)

        logging.debug(f"Loading pair id {pair_id}: {pair.get('pid1')}-{pair.get('pid2')}")

        try:
            align = BinaryAlignment(filename)
        except Exception:
            logging.warning(f"Failed to load pair {pair_id} '{pair_name}', trying to concatenate.")
            align = self.concatenate(pair, alignment_dir, downsample=downsample,
                                    downsample_to=downsample_to)
            align.Reweight_Sequence(reweighting_threshold)
            align.To_Zarr(filename, overwrite=True)

        if getattr(align, 'weights', None) is None or align.Beff is None:
            align.Reweight_Sequence(reweighting_threshold)

        B, N = align.matrix.shape
        # RAW reweighting vector
        W = np.asarray(align.weights, dtype=np.float64)
        Beff = float(align.Beff)
        q = 21

        # Hand the alignment over as int8 in residue-major (N, M) order
        Zt = np.ascontiguousarray(align.matrix.T, dtype=np.int8)   # (N, M)
        msa_size = Zt.nbytes
        w_size = W.nbytes
        logging.debug(
            f"Pair {pair_id} '{pair_name}': MSA {Zt.shape} int8 ({msa_size} B), "
            f"W {W.shape} ({w_size} B)")
        try:
            msa_shm = shared_memory.SharedMemory(
                name=self.SHM_PREFIX + pair_name, create=True, size=msa_size + w_size)
        except FileExistsError:
            logging.warning(f"Shared memory for pair {pair_id} '{pair_name}' already exists; reusing it.")
            msa_shm = shared_memory.SharedMemory(name=self.SHM_PREFIX + pair_name)
        msa_arr = np.ndarray(Zt.shape, dtype=np.int8, buffer=msa_shm.buf)
        w_arr = np.ndarray(W.shape, dtype=np.float64, buffer=msa_shm.buf[msa_size:])
        np.copyto(msa_arr, Zt)
        np.copyto(w_arr, W)
        del Zt

        params = {
            "name": pair_name,
            "pair_id": pair_id,
            "B": B,
            "N": N,
            "q": q,
            "Beff": Beff,
            "output_dir": output_dir,
            "file_hash": file_hash,
        }
        return params

    def infer(self, params):
        q = params["q"]
        N = params["N"]
        B = params["B"]
        msa_name = params["name"]
        NQ = N * (q - 1)

        # (N, N) APC-corrected score as float32
        try:
            score_shm = shared_memory.SharedMemory(
                name=self.SHM_PREFIX + msa_name + '_J', create=True,
                size=N * N * 4)
        except FileExistsError:
            pass  # already created; nothing to do

        n_threads = max(1, int(getattr(self, 'n_threads', 1)))
        cmd = [self.gauss_infer_exe, msa_name, str(B), str(N), str(q),
               str(self.pseudocount), str(n_threads)]

        # gauss_infer inverts the covariance through a threaded LAPACK
        # avoid to oversubscribe the machine
        env = dict(os.environ)
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env[var] = str(n_threads)

        logging.debug(f"Running GaussDCA inference with cmd {' '.join(cmd)}")
        rv = subprocess.run(cmd, env=env)
        return rv.returncode

    def collect_results(self, returncode, params):
        N = params["N"]
        q = params["q"]
        NQ = N * (q - 1)
        output_dir = params["output_dir"]
        msa_name = params["name"]
        file_hash = params.get("file_hash")
        filename, _ = get_hashed_file_path(msa_name, file_hash, output_dir)

        if returncode != 0:
            try:
                J_shm = shared_memory.SharedMemory(name=self.SHM_PREFIX + msa_name + '_J', size=N * N * 4)
                J_shm.close()
                J_shm.unlink()
                shm = shared_memory.SharedMemory(name=self.SHM_PREFIX + msa_name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            raise RuntimeError(
                f"GaussDCA inference failed for pair {params['pair_id']} '{msa_name}' (rc={returncode})"
            )

        # gauss_infer already applied the Ising gauge, the gap-excluded
        # Frobenius norm and the APC, so this is the finished score matrix.
        J_shm = shared_memory.SharedMemory(name=self.SHM_PREFIX + msa_name + '_J', size=N * N * 4)
        score = np.ndarray((N, N), dtype=np.float32, buffer=J_shm.buf).astype(np.float16)
        logging.info(f"Writing coupling score for pair {params['pair_id']} '{msa_name}' to alignment zarr")
        grp = zarr.open_group(store=filename)
        z = grp.require_array(name=self.couple_table, shape=score.shape, dtype='float16', overwrite=True)
        z[:] = score

        result = {
            'id': params["pair_id"],
            'effnumber': params["Beff"],
            'length': int(params["N"]),
            'number': int(params["B"]),
            'file_hash': file_hash,
        }
        try:
            J_shm.close()
            J_shm.unlink()
            shm = shared_memory.SharedMemory(name=self.SHM_PREFIX + msa_name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        return result

    def calc_graph(self, pair, priority=0, **kwargs):
        # not to over subscribe
        n_threads = max(1, int(getattr(self, 'n_threads', 1)))
        compute_slots = min(n_threads, int(getattr(self, 'compute_threads', n_threads)))

        N = pair.get('length')
        if not N:
            length1 = self.db_conn.execute(
                f"select length from {quote_identifier(self.align_table)} where pid = ?",
                (pair.get('pid1'),)).fetchone()[0]
            length2 = self.db_conn.execute(
                f"select length from {quote_identifier(self.align_table)} where pid = ?",
                (pair.get('pid2'),)).fetchone()[0]
            N = length1 + length2
        per_pair_mb = max(1.0, min(self._pair_memory_mb(N), self.mem_budget_mb))

        with annotate(resources={'loader': 1}, priority=0):
            params = delayed(self.load_pair)(pair, **kwargs)
        with annotate(resources={'compute': compute_slots, 'memory_mb': per_pair_mb}, priority=priority):
            rc = delayed(self.infer)(params)
            out_d = delayed(self.collect_results)(rc, params)
        return out_d


EXECUTORS = {
    'plm': (PLMCouplingExecutor, PLMCouplingExecutorGPU),
    'gauss': (GaussCouplingExecutor, GaussCouplingExecutor),
}


if __name__ == '__main__':
    dask.config.set({
        'distributed.worker.memory.spill': False,
        'distributed.worker.memory.target': False,
        'distributed.worker.memory.pause': 0.8,
        'distributed.worker.memory.terminate': False,
        'distributed.comm.timeouts.connect': '60s',
        'distributed.comm.timeouts.tcp': '120s',
        'distributed.scheduler.worker-ttl': None,
        'distributed.scheduler.allowed-failures': 0,
        'distributed.worker.lifetime.duration': None,
        'distributed.worker.lifetime.restart': False,
    })

    parser = argparse.ArgumentParser(description='Coupling calculation executor')
    parser.add_argument('--config', required=True, help='Project config YAML')
    args = parser.parse_args()

    cfg = load_project_and_system_config(args.config)
    setup_logging(cfg)

    method = cfg.get('couple', {}).get('method', 'plm')
    executor_pair = EXECUTORS.get(method)
    if executor_pair is None:
        raise SystemExit(f"Unknown coupling method: {method}")

    cpu_cls, gpu_cls = executor_pair
    if gpu_is_available():
        ExecutorClass = gpu_cls
        import cupy as cp
    else:
        ExecutorClass = cpu_cls
    logging.info(f"Using executor: {ExecutorClass.__name__}")

    executor = ExecutorClass(cfg)
    executor.run()

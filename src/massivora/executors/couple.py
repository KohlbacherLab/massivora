import argparse
import concurrent.futures
import logging
import multiprocessing as mp
import os
import queue
import signal
import sqlite3
import subprocess
import threading
import time
from functools import lru_cache
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import shared_memory

import dask
import numpy as np
import psutil
import zarr
from dask import annotate, delayed
from dask.distributed import (Client, Nanny, Scheduler, SpecCluster,
                              TimeoutError, WorkerPlugin, wait)

from massivora import cpp_bindings
from massivora.alignment import (BinaryAlignment, SharedSourceError,
                                 check_shared_source_strategy,
                                 resolve_shared_source)
from massivora.config import load_project_and_system_config
from massivora.db import (STATUS, apply_cascade, compute_file_hash, connect_db_rw,
                          ensure_columns, get_db_path, get_hashed_file_path,
                          get_table_names, quote_identifier)
from massivora.utils import (SHM_PREFIX, ShardWriter, compute_id_range,
                             gpu_is_available, massivora_pkg_dir,
                             saved_columns_to_bitmask, setup_logging,
                             visible_gpu_devices, worker_id)
from massivora.analyzer import (BuiltinGaussCalibrator, BuiltinPLMCalibrator,
                                ComplexAnalyzer)

@lru_cache(maxsize=8)
def _monomer(path):
    """A monomer alignment, cached per process: consecutive pairs mostly share
    their first protein. Concatenation copies the rows, so it stays unchanged."""
    return BinaryAlignment(path)


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
        # What concatenate() does with a pair that glues a sequence to itself
        self.shared_source_strategy = check_shared_source_strategy(
            config.get('concatenate').get('shared-source_strategy', 'skip'))

        coupling_cfg = config.get('couple')
        # False: pairs concatenated here are coupled from memory and not
        # stored (the fused screen mode, without the concatenate stage)
        self.keep_concatenated = bool(coupling_cfg.get('keep_concatenated', True))
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

        # Every finished pair's CN score goes into msv shards named after the
        # couplings table, in the couplings output directory. These two options
        # are about the Jij tensor instead, which nothing stores yet.
        self.keep_coupling_matrix = bool(coupling_cfg.get('keep_coupling_matrix', False))
        self.coupling_matrix_keep_option = str(
            coupling_cfg.get('coupling_matrix_keep_option', 'all')).lower()
        if self.keep_coupling_matrix:
            logging.warning("couple.keep_coupling_matrix is set, but keeping the Jij tensor is "
                            "not implemented yet; only the CN score matrices are stored")
        self._score_writer = None

        self.job_id = worker_id()
        self.plm_opt_exe = None

        ensure_columns(self._config)
        apply_cascade(self._config)

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
        """Exclude db_conn and the shard writer from pickling (workers use neither)."""
        state = self.__dict__.copy()
        state['db_conn'] = None
        state['_score_writer'] = None
        return state

    def __setstate__(self, state):
        """Restore state without DB connection."""
        self.__dict__.update(state)

    def concatenate(self, pair, alignment_dir, downsample=False,
                    downsample_to=None):
        """
        Concatenate a pair's monomer alignments on the fly.

        Raises
        ------
        `SharedSourceError`
            If the pair is shared-source and concatenate.shared-source_strategy
            is 'skip' (or 'fix' left no species); the coordinator marks the
            pair FAILED instead of retrying it
        """
        pair_name = pair.get('name')
        pid1 = pair.get('pid1')
        pid2 = pair.get('pid2')
        if os.path.exists(os.path.join(alignment_dir, pair_name)):
            return BinaryAlignment(os.path.join(alignment_dir, pair_name))
        concatenated = _monomer(os.path.join(alignment_dir, pid1)) + \
            _monomer(os.path.join(alignment_dir, pid2))
        resolve_shared_source(concatenated, self.shared_source_strategy, pair_name)
        if downsample:
            concatenated.Downsample_Randomly(to=downsample_to)
        concatenated.Mask_Gap_Columns(self._config.get('align').get('col_gap_threshold', 0.5))
        return concatenated

    def _load_or_concatenate(self, pair, filename, kwargs):
        """The pair's stored alignment, or one concatenated here from the
        monomers, which is stored unless couple.keep_concatenated is false."""
        if os.path.exists(filename):
            try:
                return BinaryAlignment(filename)
            except Exception:
                logging.warning(f"Failed to load pair {pair.get('id')} '{pair.get('name')}', "
                                "concatenating it again.")
        align = self.concatenate(pair, kwargs.get('alignment_dir'),
                                 downsample=kwargs.get('downsample', False),
                                 downsample_to=kwargs.get('downsample_to'))
        if self.keep_concatenated:
            align.To_Zarr(filename, overwrite=True)
        return align

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

    def _db_apply_results(self, done_updates, fail_updates, skipped=()):
        """
        Commit finished pairs, count an attempt for failed ones (they are
        claimed again until their attempts run out and they end FAILED), and
        mark `skipped` ones FAILED at once: those are shared-source pairs,
        which another attempt would not change.
        """
        if not done_updates and not fail_updates and not skipped:
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
                        pcontact = ?,
                        neffoverL = ?,
                        log_lr_raw = ?,
                        confidence_flag = ?,
                        high_precision_hit = ?,
                        file_hash = ?,
                        score_file_id = ?,
                        score_offset = ?,
                        score_len = ?,
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
                            x.get('pcontact', 0),
                            x.get('neffoverL', 0),
                            x.get('log_lr_raw', 0),
                            x.get('confidence_flag', False),
                            x.get('high_precision_hit', False),
                            x.get('file_hash'),
                            x.get('score_file_id'),
                            x.get('score_offset'),
                            x.get('score_len'),
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

            if skipped:
                cursor.executemany(
                    f"""
                    UPDATE {table}
                    SET status = ?,
                        claimed_at = NULL,
                        job_id = NULL
                    WHERE id = ?
                    """,
                    [(STATUS['FAILED'], i) for i in skipped],
                )
                n += len(skipped)

            self.db_conn.commit()
            return n
        except Exception as e:
            logging.error(f"Failed to apply results to DB for job_id={self.job_id}: {e}")
            self.db_conn.rollback()
            raise

    def _db_unclaim(self, ids):
        """Hand claimed pairs back unrun, undoing the claim without spending an
        attempt -- the rule massiveilance.fix_zombie_tasks applies to a dead job."""
        if not ids:
            return 0
        table = quote_identifier(self.couple_table)
        try:
            self.db_conn.executemany(
                f"""
                UPDATE {table}
                SET status = CASE status WHEN :running THEN :noopt
                                         WHEN :retrying THEN :pending
                                         ELSE status END,
                    claimed_at = NULL,
                    job_id = NULL
                WHERE id = :id AND job_id = :job_id
                """,
                [{'running': STATUS['RUNNING'], 'noopt': STATUS['NOOPT'],
                  'retrying': STATUS['RETRYING'], 'pending': STATUS['PENDING'],
                  'id': i, 'job_id': str(self.job_id)} for i in ids],
            )
            self.db_conn.commit()
            return len(ids)
        except Exception as e:
            logging.error(f"Failed to hand back claimed pairs for job_id={self.job_id}: {e}")
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

    # -- Score storage --

    def _score_fields(self, score, params):
        """
        What a result carries so the coordinator can store the pair's score.

        Runs where the score is computed. Nothing is written here: only the
        coordinator writes shards, so the shard it fsyncs and the rows it commits
        always agree.

        Parameters
        ----------
        `score` — np.ndarray
            The pair's full-width CN score matrix
        `params` — dict
            The pair's parameters, holding `name`, `saved_columns` and `lengthes`

        Returns
        -------
        `dict`
            `name`, `score` and `bitmask`
        """
        return {
            'name': params['name'],
            'score': np.asarray(score, dtype=np.float16),
            'bitmask': saved_columns_to_bitmask(params['saved_columns'], params['lengthes']),
        }

    def _store_score(self, result):
        """
        Append the score a result carries to this job's shards and record where it went.

        Parameters
        ----------
        `result` — dict
            A finished task's result, as built with `_score_fields()`

        Returns
        -------
        `dict`
            The result with `score` and `bitmask` replaced by `score_file_id`,
            `score_offset` and `score_len`; unchanged when it carries no score
        """
        score = result.pop('score', None)
        bitmask = result.pop('bitmask', b'')
        if score is None:
            return result
        if not np.array_equal(score, score.T, equal_nan=True):
            upper = np.triu_indices(score.shape[0], 1)
            change = np.abs(score[upper].astype(np.float32) - score.T[upper].astype(np.float32))
            logging.warning(
                f"CN score matrix for pair {result.get('id')} '{result['name']}' is not exactly "
                f"symmetric; its shard record keeps the upper triangle (the lower one differs "
                f"by up to {float(np.nan_to_num(change, nan=np.inf).max()):.3g})")
        if self._score_writer is None:
            self._score_writer = ShardWriter(self.output_dir, self.couple_table)
        file_id, offset, length = self._score_writer.append(result['name'], score, bitmask)
        result.update(score_file_id=file_id, score_offset=offset, score_len=length)
        logging.info(f"Stored coupling score for pair {result.get('id')} '{result['name']}' in "
                     f"{self.couple_table} shard {file_id} at offset {offset}")
        return result

    def _flush_scores(self):
        """Make appended scores durable. Call before committing rows that point at them."""
        if self._score_writer is not None:
            self._score_writer.flush()

    def _close_scores(self):
        """Flush and release this job's shard."""
        if self._score_writer is not None:
            self._score_writer.close()
            self._score_writer = None

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
    def _expand_score(score, saved_columns, lengthes):
        """Scatter a reduced score matrix back to the full alignment width.

        fast_approximation optimizes only the columns the gap mask keeps, but
        the couplings array has to stay the size of the complete alignment so
        the analyzers can go on indexing it by residue number.

        The Zarr archive still holds the whole alignment, so its own metadata
        says where a reduced column belongs: `saved_columns` holds one list of
        monomer-local indices per monomer and `lengthes` the monomers' original
        widths, so a column's global index is its local index plus the offset
        of its monomer. Positions that were never optimized are left at zero.

        Parameters
        ----------
        `score` — np.ndarray
            The (n, n) score over the reduced columns
        `saved_columns` — list of list of int
            Per-monomer local indices of the kept columns
        `lengthes` — list of int
            Original width of each monomer

        Returns
        -------
        `np.ndarray`
            The score at the full alignment width, or `score` itself when the
            metadata describes no reduction

        Raises
        ------
        `RuntimeError`
            If the mask keeps a different number of columns than were
            optimized, which would place every value in the wrong cell
        """
        if not saved_columns or not lengthes or len(saved_columns) != len(lengthes):
            logging.warning(
                "saved_columns does not describe this alignment; "
                "writing the score as it is")
            return score

        columns = []
        offset = 0
        for local_cols, length in zip(saved_columns, lengthes):
            columns.extend(offset + int(col) for col in local_cols)
            offset += int(length)
        n_full = offset

        if not columns or len(columns) == n_full:
            return score  # the mask keeps every column; nothing was dropped
        if len(columns) != score.shape[0]:
            raise RuntimeError(
                f"saved_columns keeps {len(columns)} columns but the score covers "
                f"{score.shape[0]}; refusing to write a misaligned coupling matrix")

        full = np.zeros((n_full, n_full), dtype=score.dtype)
        full[np.ix_(columns, columns)] = score
        return full

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
            The pairwise coupling matrix, exactly symmetric
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
        FN = (FN + FN.T) / 2

        # Average Product Correction of Frobenius norm
        mean = np.sum(FN, axis=1, keepdims=True) / (N - 1)
        total_mean = np.sum(FN) / (N * (N - 1))
        FN_APC = FN - (mean @ mean.T) / total_mean

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
            # restore logging when worker restarts
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
            self._close_scores()
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
        skipped_ids = []    # shared-source pairs, marked FAILED without a retry

        last_commit = float(time.time())
        terminating = False

        def flush(force=False):
            nonlocal done_updates, fail_ids, skipped_ids, last_commit
            if not done_updates and not fail_ids and not skipped_ids:
                return 0

            if not force:
                if (len(done_updates) + len(fail_ids) + len(skipped_ids)) < int(commit_every_tasks) and (time.time() - last_commit) < float(commit_every_seconds):
                    return 0

            self._flush_scores()
            n = self._db_apply_results(done_updates, fail_ids, skipped_ids)
            self._checkpoint_wal()
            self._check_shared_memory(inflight)
            done_updates = []
            fail_ids = []
            skipped_ids = []
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
                    if not rows and fail_ids:
                        # Buffered failures aren't claimable until committed;
                        # commit them and look again before giving up.
                        flush(force=True)
                        continue
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
                        res = self._store_score(future.result())
                    except SharedSourceError as e:
                        logging.warning(f"Task {task_id}: {e}; marking it FAILED")
                        skipped_ids.append(task_id)
                    except Exception:
                        logging.exception(f"Task {task_id} failed")
                        fail_ids.append(task_id)
                    else:
                        done_updates.append(res)
                        future.release()

            # Failed pairs only become claimable (PENDING) once committed. If the
            # queue has already run dry, commit them now and resume claiming so
            # this job retries them instead of leaving them stranded at PENDING.
            if fail_ids and no_more_to_claim and not terminating:
                flush(force=True)
                no_more_to_claim = False

            # periodic flush
            flush(force=False)

        # final flush
        flush(force=True)
        self._close_scores()

        if terminating:
            self.close_db_connection()
            raise SystemExit(143)


# ---------------------------------------------------------------------------
# CPU PLM executor
# ---------------------------------------------------------------------------

class PLMCouplingExecutor(BaseCouplingExecutor, BuiltinPLMCalibrator):
    def __init__(self, config):
        super().__init__(config)
        self.plm_opt_exe = os.path.join(massivora_pkg_dir(), 'bin', 'plm_opt_site')
        self.fast_approximation = bool(config.get('couple').get('fast_approximation', False))

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

        q = int(align.matrix.max()) + 1

        # Optimize only the columns the gap mask keeps
        if self.fast_approximation:
            align.Mask_Gap_Columns(self._config.get('align').get('col_gap_threshold',0.5))
            align.To_Zarr(filename, overwrite=True)
            align.Drop_Masked_Gap_Columns()

        MSA = np.ascontiguousarray(align.matrix, dtype=np.int8)
        B, N = MSA.shape
        W = np.array(align.weights, dtype=np.float64)
        Beff = float(align.Beff)
        W /= Beff

        # Create MSA shared memory
        msa_size = MSA.nbytes
        w_offset = (msa_size + 7) & ~7
        w_size = W.nbytes
        logging.debug(f"MSA size (bytes): {msa_size}, Weights size (bytes): {w_size}")
        logging.debug(f"MSA dimensions: {MSA.shape}, Weights length: {W.shape}")
        msa_shm = shared_memory.SharedMemory(name=self.SHM_PREFIX +pair_name, create=True, size=w_offset+w_size)
        msa_arr = np.ndarray(MSA.shape, dtype=np.int8, buffer=msa_shm.buf)
        w_arr = np.ndarray(W.shape, dtype=np.float64, buffer=msa_shm.buf[w_offset:])
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
            "file_hash": file_hash,
            "saved_columns": list(align.saved_columns),
            "lengthes": list(align.lengthes),
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
        N = params["N"]
        q = params["q"]

        J_shm = shared_memory.SharedMemory(name=self.SHM_PREFIX +params["name"]+'_J', size=N*N*q*q*8)
        J = np.ndarray((N*q*q, N), dtype=np.float64, buffer=J_shm.buf)

        score = self.compute_score(J, q=q).astype(np.float16)
        if self.fast_approximation:
            # Expand the shrinked score matrix to its original size
            analyzer = ComplexAnalyzer(score, saved_columns=[list(range(len(params['saved_columns'][0]))),
                                                             list(range(len(params['saved_columns'][1])))],
                                        chainA_length=len(params['saved_columns'][0]),
                                        EC_threshold=0)
            score = self._expand_score(score, params['saved_columns'],
                                      params['lengthes'])
        n_full = int(score.shape[0])

        analyzer = ComplexAnalyzer(score, saved_columns=params['saved_columns'],
                                       chainA_length=params['lengthes'][0],
                                       EC_threshold=0)
        calibrated = self.Calibrate(analyzer, params["Beff"])

        result = {
            'id': params["pair_id"],
            'effnumber': params["Beff"],
            'length': n_full,
            'number': int(params["B"]),
            'file_hash': params.get("file_hash"),
            **calibrated,
            **self._score_fields(score, params),
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
# GaussDCA executor
# ---------------------------------------------------------------------------

class GaussCouplingExecutor(BaseCouplingExecutor, BuiltinGaussCalibrator):
    def __init__(self, config):
        super().__init__(config)
        coupling_cfg = config.get('couple')
        self.pseudocount = float(coupling_cfg.get('pseudocount', 0.8))
        threads_cfg = coupling_cfg.get('gauss_threads', 2)
        self.auto_threads = (isinstance(threads_cfg, str)
                             and threads_cfg.strip().lower() == 'auto')
        self.n_threads = None if self.auto_threads else max(1, int(threads_cfg))
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
            if info.get('ilp64') == '0':
                logging.warning(
                    "%s is not an ILP64 build; pairs wider than 2317 columns will "
                    "fall back to a serial inverse and use ~3.7x the memory "
                    "reserved for them. Reinstall massivora against an ILP64 "
                    "openblas.", self.gauss_infer_exe)
            if configured:
                return int(configured)
            if 'cov_bytes' in info:
                return int(info['cov_bytes'])
        except Exception as e:
            logging.warning(f"Could not probe {self.gauss_infer_exe} for its precision "
                            f"({e}); assuming float64. A single-precision build will "
                            f"then be over-reserved by 2x.")
        return 8

    # Width tiers for gauss_threads: auto
    _THREAD_TIERS = ((500, 1), (1000, 2), (1500, 4), (2000, 8))
    _THREAD_TIER_MAX = 16

    def _threads_for(self, N, budget):
        """Threads to give one gauss_infer run for an alignment of width N."""
        if not self.auto_threads:
            want = self.n_threads
        else:
            want = self._THREAD_TIER_MAX
            for upper, tier in self._THREAD_TIERS:
                if N <= upper:
                    want = tier
                    break
        return max(1, min(int(want), int(budget)))

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
        self.reweight_threads = 4

        logging.info(
            f"Creating GaussDCA cluster: 1 worker, {total_cpus} task slots, "
            f"memory budget {self.mem_budget_mb:.0f} MB"
        )

        worker_spec = {
            'worker': {
                'cls': Nanny,
                'options': {
                    'nthreads': total_cpus,
                    'resources': {
                        'compute': total_cpus,
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
        client.wait_for_workers(1, timeout=60)
        return cluster, client

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
            align.Reweight_Sequence(reweighting_threshold,
                                    n_threads=getattr(self, 'reweight_threads', 0))
            align.To_Zarr(filename, overwrite=True)

        if getattr(align, 'weights', None) is None or align.Beff is None:
            align.Reweight_Sequence(reweighting_threshold,
                                    n_threads=getattr(self, 'reweight_threads', 0))

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
            "saved_columns": list(align.saved_columns),
            "lengthes": list(align.lengthes),
        }
        return params

    def infer(self, params, n_threads=1):
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

        # Must match the 'compute' slots calc_graph reserved for this task,
        # otherwise the admission control and the process disagree.
        n_threads = max(1, int(n_threads))
        cmd = [self.gauss_infer_exe, msa_name, str(B), str(N), str(q),
               str(self.pseudocount), str(n_threads)]

        # gauss_infer inverts the covariance through a threaded LAPACK
        # avoid to oversubscribe the machine
        env = dict(os.environ)
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
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

        analyzer = ComplexAnalyzer(score, saved_columns=params['saved_columns'],
                                       chainA_length=params['lengthes'][0],
                                       EC_threshold=0)
        calibrated = self.Calibrate(analyzer, params["Beff"])

        result = {
            'id': params["pair_id"],
            'effnumber': params["Beff"],
            'length': int(params["N"]),
            'number': int(params["B"]),
            'file_hash': params.get("file_hash"),
            **calibrated,
            **self._score_fields(score, params),
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

    _COUPLE_PRIORITY = 1 << 20

    def calc_graph(self, pair, priority=0, **kwargs):
        budget = mp.cpu_count()

        N = pair.get('length')
        if not N:
            length1 = self.db_conn.execute(
                f"select length from {quote_identifier(self.align_table)} where pid = ?",
                (pair.get('pid1'),)).fetchone()[0]
            length2 = self.db_conn.execute(
                f"select length from {quote_identifier(self.align_table)} where pid = ?",
                (pair.get('pid2'),)).fetchone()[0]
            N = length1 + length2
        n_threads = self._threads_for(N, budget)
        per_pair_mb = max(1.0, min(self._pair_memory_mb(N), self.mem_budget_mb))

        # estimate the memory load of pairs
        b_cap = int(self.downsample_to if self.downsample
                    else (pair.get('number') or self.downsample_to))
        load_mb = max(64.0, min(6.0 * b_cap * int(N) / 1e6, self.mem_budget_mb))

        with annotate(resources={'compute': self.reweight_threads, 'memory_mb': load_mb},
                      priority=priority):
            params = delayed(self.load_pair)(pair, **kwargs)
        with annotate(resources={'compute': n_threads, 'memory_mb': per_pair_mb},
                      priority=priority + self._COUPLE_PRIORITY):
            rc = delayed(self.infer)(params, n_threads)
        with annotate(resources={'compute': 1, 'memory_mb': 1.0},
                      priority=priority + 2 * self._COUPLE_PRIORITY):
            out_d = delayed(self.collect_results)(rc, params)
        return out_d


# ---------------------------------------------------------------------------
# GaussDCA executor GPU
# ---------------------------------------------------------------------------

class GpuAdmission(object):
    """
    Lets pairs onto the GPUs. A pair starts on a GPU only when that GPU has both
    an idle session and room for the pair's device memory, and takes the two at
    once: holding memory while queueing for a session, or a session while
    queueing for memory, would leave the GPU idle with work ready.

    Pairs go in arrival order, except that a later pair which fits may start
    while the oldest one waits for memory to drain -- the GPU keeps working
    meanwhile. Once the oldest has waited `patience` seconds nothing overtakes
    it any more, so a large pair cannot be starved by a stream of small ones.
    """

    def __init__(self, budgets, sessions, patience=30.0):
        self.budgets = [int(b) for b in budgets]
        self._free = list(self.budgets)
        self._idle = [int(sessions)] * len(self.budgets)
        self.patience = float(patience)
        self._cond = threading.Condition()
        self._waiting = []                 # (ticket, since), oldest first

    def acquire(self, nbytes):
        """Wait for a GPU with an idle session and `nbytes` to spare; take both and return its ordinal."""
        if nbytes > max(self.budgets):
            raise ValueError(f"{nbytes} bytes exceed every GPU's budget {self.budgets}")
        me = (object(), time.monotonic())
        with self._cond:
            self._waiting.append(me)
            try:
                while True:
                    ordinal = self._fit(nbytes, me)
                    if ordinal is not None:
                        self._free[ordinal] -= nbytes
                        self._idle[ordinal] -= 1
                        return ordinal
                    self._cond.wait()
            finally:
                self._waiting.remove(me)
                self._cond.notify_all()

    def _fit(self, nbytes, me):
        oldest = self._waiting[0]
        if oldest is not me and time.monotonic() - oldest[1] > self.patience:
            return None
        fits = [o for o in range(len(self._free))
                if self._idle[o] and self._free[o] >= nbytes]
        return max(fits, key=self._free.__getitem__) if fits else None

    def release(self, ordinal, nbytes):
        with self._cond:
            self._free[ordinal] += nbytes
            self._idle[ordinal] += 1
            self._cond.notify_all()


class StageClock(object):
    """Where the GPU executor's pairs spend their time, logged every `every` seconds."""

    STAGES = (('load', 'load'), ('wait', 'waiting for a GPU'), ('gpu', 'on the GPU'),
              ('analysis', 'analysis'))

    def __init__(self, sessions, every=60.0, label='GPU'):
        self.sessions = int(sessions)
        self.every = float(every)
        self.label = label
        self._lock = threading.Lock()
        self._reset(time.monotonic())

    def _reset(self, now):
        self._since = now
        self._pairs = 0
        self._sum = dict.fromkeys((name for name, _ in self.STAGES), 0.0)

    def add(self, **seconds):
        """Count one finished pair and the seconds it spent in each stage."""
        with self._lock:
            self._pairs += 1
            for name, value in seconds.items():
                self._sum[name] += value
            now = time.monotonic()
            if now - self._since < self.every:
                return
            elapsed, pairs, total = now - self._since, self._pairs, dict(self._sum)
            self._reset(now)
        logging.info(
            "%s: %d pairs in %.0f s (%.1f/s); per pair %s; sessions busy %.0f%%",
            self.label, pairs, elapsed, pairs / elapsed,
            ", ".join(f"{label} {total[name] / pairs:.3f} s" for name, label in self.STAGES),
            100.0 * total['gpu'] / (elapsed * self.sessions))


# Loading an alignment and analysing its score are mostly Python, and in the
# coordinator's threads they all queue on one GIL with the GPU dispatch, which
# then waits on them. The GPU executor runs them in worker processes instead.
_HOST = None


def _host_init(cls, state):
    """Rebuild the executor in a host process without __init__, which opens the job database."""
    global _HOST
    _HOST = cls.__new__(cls)
    _HOST.__setstate__(state)
    setup_logging(state['_config'])

    # A coordinator killed outright (SIGKILL) never shuts this process down, and
    # it would wait on the call queue forever. Leave once it is gone.
    coordinator = os.getppid()

    def leave_with_coordinator():
        while os.getppid() == coordinator:
            time.sleep(5)
        os._exit(0)

    threading.Thread(target=leave_with_coordinator, daemon=True).start()


class PairArrays(object):
    """A pair's alignment and weights, held until the GPU has its own copy."""

    def __init__(self, MSA, W, shm=None):
        self.MSA, self.W, self._shm = MSA, W, shm

    @classmethod
    def attach(cls, name, w_shape, msa_shape):
        """The arrays a host process left in the shared memory block `name`;
        `w_shape` is None for a pair whose weights the GPU is to compute."""
        shm = shared_memory.SharedMemory(name=name)
        W = None if w_shape is None else np.ndarray(w_shape, dtype=np.float64, buffer=shm.buf)
        offset = 0 if W is None else W.nbytes
        return cls(np.ndarray(msa_shape, dtype=np.int8, buffer=shm.buf, offset=offset),
                   W, shm)

    def release(self):
        self.MSA = self.W = None
        shm, self._shm = self._shm, None
        if shm is None:
            return
        shm.unlink()                    # the name goes now, the memory once unmapped
        try:
            shm.close()
        except BufferError:
            # The traceback of a failed score still holds views of it; the block
            # closes itself when collected, after that traceback is gone.
            self._dangling = shm


def _host_load(pair, kwargs):
    """Load a pair and leave its arrays in shared memory: sent back through the
    pool's pipe they would be pickled, copied and unpickled by one thread of the
    coordinator, which caps how many pairs a second reach the GPUs."""
    MSA, W, params = _HOST.load_pair_arrays(pair, **kwargs)
    name = f"{SHM_PREFIX}g{os.getpid()}_{pair['id']}"
    w_bytes = 0 if W is None else W.nbytes
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=w_bytes + MSA.nbytes)
    except FileExistsError:                 # left by a run that died holding this pair
        stale = shared_memory.SharedMemory(name=name)
        stale.close()
        stale.unlink()
        shm = shared_memory.SharedMemory(name=name, create=True, size=w_bytes + MSA.nbytes)
    try:
        if W is not None:
            np.ndarray(W.shape, dtype=np.float64, buffer=shm.buf)[:] = W
        np.ndarray(MSA.shape, dtype=np.int8, buffer=shm.buf, offset=w_bytes)[:] = MSA
    except BaseException:
        shm.close()
        shm.unlink()
        raise
    shm.close()
    return name, None if W is None else W.shape, MSA.shape, params


def _host_collect(score, params):
    return _HOST.collect_score(score, params)


class SessionGPUExecutor(BaseCouplingExecutor):
    """
    Runs pairs on device sessions in threads of this process instead of Dask
    workers: each GPU gets `couple.gpu_sessions` sessions of the subclass's
    CUDA module, a pair starts once GpuAdmission finds a GPU with an idle
    session and room for its device memory, and loading and analysis run in
    host processes. A session keeps its CUDA context, handles and staging
    between pairs and takes each pair's device memory from the device's pool,
    so what the pairs in flight hold is what admission reserved for them.

    Subclasses provide the CUDA module and how a pair is sized, loaded, run on
    a session and analysed.
    """

    GPU_LABEL = 'GPU'

    def __init__(self, config):
        super().__init__(config)
        coupling_cfg = config.get('couple')

        self.gpu_sessions = int(coupling_cfg.get('gpu_sessions', 4))
        self.gpu_workers = int(coupling_cfg.get('gpu_workers', 16))
        # Share of each GPU's free memory the pairs in flight may hold between them;
        # the rest is for the CUDA libraries' own workspaces.
        self.gpu_memory_limit = float(coupling_cfg.get('gpu_memory_limit', 0.9))
        # Processes that load alignments and analyse scores; 0 keeps that in
        # the coordinator's threads.
        host = coupling_cfg.get('gpu_host_processes', 'auto')
        if isinstance(host, str) and host.strip().lower() == 'auto':
            self.gpu_host_processes = min(8, os.cpu_count() or 1)
        else:
            self.gpu_host_processes = max(0, int(host))

    # -- what a subclass provides ----------------------------------------------

    def _gpu_module(self):
        """The CUDA module: Session, DeviceLostError and device_memory."""
        raise NotImplementedError

    def _new_session(self, gpu, ordinal):
        return gpu.Session(ordinal)

    def _pair_bytes(self, gpu, params, budget):
        """Device memory the pair reserves. It may adapt how the pair runs
        (recording that in params) to fit the largest GPU budget, `budget`."""
        raise NotImplementedError

    def _run_on_session(self, sess, MSA, W, params, kwargs):
        """The pair's (N, N) score, computed on `sess`; W is None for a pair whose
        weights are to be computed on the GPU (record them in params)."""
        raise NotImplementedError

    def load_pair_arrays(self, pair, **kwargs):
        raise NotImplementedError

    def collect_score(self, score, params):
        raise NotImplementedError

    def _keep_gpu_weights(self, params):
        """A pair reweighted on the GPU: keep the weights with the stored
        alignment, where the CPU path's To_Zarr puts them, so a later run
        reuses them and massiveilance recovers this pair's Beff from them."""
        weights = params.pop('weights', None)
        if weights is None:
            return
        filename, _ = get_hashed_file_path(params['name'], params['file_hash'],
                                           self.output_dir)
        if not os.path.exists(filename):        # coupled from memory, not stored
            return
        zarr.open_group(store=filename)['align'].attrs.update(
            {'weights': np.asarray(weights).tolist(), 'Beff': params['Beff']})

    # -- dispatch -------------------------------------------------------------

    def _claimed_pairs(self, id_range=None, stop=None):
        """Stream claimed rows; the work queue bounds how far ahead we run.
        Once `stop` is set no new batch is claimed, but the rows of the current
        one are still yielded, so whoever claimed them can hand them back."""
        while stop is None or not stop.is_set():
            rows = self._claim_batch(self.job_id, self.claim_batch_size,
                                     id_range=id_range)
            if not rows:
                return
            for r in rows:
                yield {k: r[k] for k in r.keys()}

    def run(self, client=None, id_range=None):
        gpu = self._gpu_module()

        kwargs = dict(alignment_dir=self.alignment_dir,
                      output_dir=self.output_dir,
                      downsample=self.downsample,
                      downsample_to=self.downsample_to,
                      reweighting_threshold=self.reweighting_threshold)

        devices = visible_gpu_devices()
        sessions = []
        for ordinal, device_id in enumerate(devices):
            logging.debug(f"{self.GPU_LABEL}: %d session(s) on ordinal %d (GPU %s)",
                          self.gpu_sessions, ordinal, device_id)
            sessions.append(queue.Queue())
            for _ in range(self.gpu_sessions):
                sessions[ordinal].put(self._new_session(gpu, ordinal))
        logging.info(f"{self.GPU_LABEL}: %d session(s) over %d GPU(s) (CUDA_VISIBLE_"
                     "DEVICES=%s), %d workers", self.gpu_sessions * len(devices),
                     len(devices), ",".join(devices), self.gpu_workers)

        n_sessions = self.gpu_sessions * len(devices)
        if self.gpu_workers < n_sessions + self.gpu_host_processes:
            logging.warning(f"{self.GPU_LABEL}: %d workers cannot keep %d sessions busy while "
                            "%d host processes load and analyse; raise couple.gpu_workers",
                            self.gpu_workers, n_sessions, self.gpu_host_processes)

        # Sessions take device memory per pair and hand it back after, so what
        # the pairs in flight hold is what they reserved here. Measured once the
        # sessions exist, so their contexts and handles are already off it.
        admission = GpuAdmission(
            (int(self.gpu_memory_limit * gpu.device_memory(o)['free_mb'] * 2**20)
             for o in range(len(devices))), self.gpu_sessions)
        logging.info(f"{self.GPU_LABEL}: device memory budget %s GiB",
                     ", ".join(f"{b / 2**30:.1f}" for b in admission.budgets))
        clock = StageClock(n_sessions, label=self.GPU_LABEL)

        # This file may be running as __main__ (massiworker runs it with
        # runpy), which a spawned process cannot import; hand it the module.
        from massivora.executors import couple as importable
        pool_lock = threading.Lock()
        pool = [None]

        def new_pool():
            return concurrent.futures.ProcessPoolExecutor(
                self.gpu_host_processes, mp_context=mp.get_context('spawn'),
                initializer=importable._host_init,
                initargs=(getattr(importable, type(self).__name__), self.__getstate__()))

        def on_host(fn, *args):
            current = pool[0]
            try:
                return current.submit(fn, *args).result()
            except BrokenProcessPool:
                # A host process died, which breaks the pool for every pair;
                # start a new one so only the pairs it was holding fail.
                with pool_lock:
                    if pool[0] is current:
                        logging.error(f"{self.GPU_LABEL}: a host process died; restarting them")
                        current.shutdown(wait=False, cancel_futures=True)
                        pool[0] = new_pool()
                raise

        if self.gpu_host_processes:
            pool[0] = new_pool()

            def load(pair):
                name, w_shape, msa_shape, params = on_host(importable._host_load, pair, kwargs)
                return PairArrays.attach(name, w_shape, msa_shape), params

            collect = lambda score, params: on_host(importable._host_collect, score, params)
        else:
            def load(pair):
                MSA, W, params = self.load_pair_arrays(pair, **kwargs)
                return PairArrays(MSA, W), params

            collect = self.collect_score

        work = queue.Queue(maxsize=self.gpu_workers * 2)
        # skipped: shared-source pairs, marked FAILED without a retry
        done_updates, fail_updates, released, skipped = [], [], [], []
        lock = threading.Lock()
        # An illegal address and its like corrupt the context every session in
        # this process shares; from then on every pair would fail and spend its
        # attempts. So stop instead, and hand back the pairs not yet run.
        lost = threading.Event()
        # Set when the device is lost or the job is told to stop: claim nothing
        # more, hand back the pairs not yet on a GPU and let those on one finish.
        halt = threading.Event()
        underestimated = threading.Event()

        def on_signal(signum, frame):
            halt.set()
            # A second signal gets the handler that was there before: it stops
            # the job at once.
            signal.signal(signum, previous_handlers.get(signum, signal.SIG_DFL))
            logging.warning(f"{self.GPU_LABEL}: received signal %d; finishing the pairs on the "
                            "GPUs and handing the rest back", signum)

        previous_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, on_signal)

        def score_pair(MSA, W, params, spent):
            """Score on the first GPU with room for the pair; None if the run halted meanwhile."""
            need = self._pair_bytes(gpu, params, max(admission.budgets))
            if need > max(admission.budgets):
                raise RuntimeError(
                    f"Pair {params['pair_id']} '{params['name']}' ({params['B']} x "
                    f"{params['N']}) needs {need / 2**30:.1f} GiB of device memory, more "
                    f"than the {max(admission.budgets) / 2**30:.1f} GiB "
                    f"couple.gpu_memory_limit allows on any GPU")
            if halt.is_set():
                return None
            t0 = time.monotonic()
            ordinal = admission.acquire(need)
            sess = sessions[ordinal].get()          # admission left one idle
            t1 = time.monotonic()
            try:
                if halt.is_set():
                    return None
                score = self._run_on_session(sess, MSA, W, params, kwargs)
                used = sess.last_pair_bytes
            except gpu.DeviceLostError:
                lost.set()
                halt.set()
                raise
            finally:
                sessions[ordinal].put(sess)
                admission.release(ordinal, need)
            spent.update(wait=t1 - t0, gpu=time.monotonic() - t1)
            if used > need and not underestimated.is_set():
                underestimated.set()
                logging.warning(
                    f"{self.GPU_LABEL}: pair %s (%d x %d) took %d MiB of device memory, more "
                    "than the %d MiB reserved for it; lower couple.gpu_memory_limit if "
                    "pairs run out of memory", params['pair_id'], params['B'],
                    params['N'], used >> 20, need >> 20)
            return score

        def worker():
            while True:
                pair = work.get()
                try:
                    if pair is None:
                        return
                    score, spent = None, {}
                    if not halt.is_set():
                        t0 = time.monotonic()
                        arrays, params = load(pair)
                        spent['load'] = time.monotonic() - t0
                        try:
                            score = score_pair(arrays.MSA, arrays.W, params, spent)
                        finally:
                            arrays.release()
                    if score is None:
                        with lock:
                            released.append(pair['id'])
                        continue
                    t0 = time.monotonic()
                    row = collect(score, params)
                    spent['analysis'] = time.monotonic() - t0
                    with lock:
                        done_updates.append(self._store_score(row))
                    clock.add(**spent)
                except SharedSourceError as e:
                    logging.warning("Pair %s: %s; marking it FAILED", pair['id'], e)
                    with lock:
                        skipped.append(pair['id'])
                except Exception:
                    logging.exception("Pair %s failed", (pair or {}).get('id'))
                    with lock:
                        fail_updates.append((pair or {}).get('id'))
                finally:
                    work.task_done()

        threads = [threading.Thread(target=worker, daemon=True)
                   for _ in range(self.gpu_workers)]
        for t in threads:
            t.start()

        def flush(force=False):
            with lock:
                if not force and len(done_updates) + len(fail_updates) + len(skipped) < 64:
                    return
                # Scores behind these rows must be on disk before the rows are committed.
                self._flush_scores()
                d, f, r, s = list(done_updates), list(fail_updates), list(released), list(skipped)
                done_updates.clear(); fail_updates.clear(); released.clear(); skipped.clear()
            if d or f or s:
                self._db_apply_results(d, f, s)
            if r:
                self._db_unclaim(r)

        try:
            while True:
                for pair in self._claimed_pairs(id_range, stop=halt):
                    work.put(pair)
                    flush(force=False)
                # Let everything dispatched finish, then commit its failures:
                # they only become claimable (PENDING) once committed, so go
                # round again to retry them rather than strand them.
                work.join()
                with lock:
                    had_failures = bool(fail_updates)
                flush(force=True)
                if halt.is_set() or not had_failures:
                    break
            for _ in threads:
                work.put(None)
            work.join()
        finally:
            for t in threads:
                t.join(timeout=30)
            if pool[0] is not None:
                pool[0].shutdown(wait=True, cancel_futures=True)
            flush(force=True)
            self._close_scores()
            self.close_db_connection()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        if lost.is_set():
            raise RuntimeError(f"{self.GPU_LABEL}: the CUDA context was lost (see the first "
                               "DeviceLostError above); the pairs not yet run were handed "
                               "back unspent. Restart the job to go on.")
        if halt.is_set():
            logging.warning(f"{self.GPU_LABEL}: stopped on a signal; the pairs not yet run were "
                            "handed back unspent")


class GaussCouplingExecutorGPU(SessionGPUExecutor, BuiltinGaussCalibrator):
    GPU_LABEL = 'GaussDCA GPU'

    def __init__(self, config):
        super().__init__(config)
        coupling_cfg = config.get('couple')
        self.pseudocount = float(coupling_cfg.get('pseudocount', 0.8))

    def _gpu_module(self):
        from massivora.cpp_bindings import gauss_cuda
        return gauss_cuda

    def _pair_bytes(self, gpu, params, budget):
        return gpu.pair_bytes(params['N'], params['B'], params['q'])

    def _run_on_session(self, sess, MSA, W, params, kwargs):
        if W is None:
            # Reweight on the GPU the pair was admitted to. The counts
            # take less device memory than the score and are done
            # before it starts, so the reservation covers both.
            m = sess.neighbour_counts(MSA, kwargs['reweighting_threshold'] * params['N'])
            W = 1.0 / m.astype(np.int64)        # exactly as Reweight_Sequence
            params['Beff'] = float(W.sum())
            params['weights'] = W
        return sess.score(MSA, W, q=params['q'], pseudocount=self.pseudocount)

    def load_pair_arrays(self, pair, **kwargs):
        pair_name = pair.get('name')
        file_hash = pair.get('file_hash') or compute_file_hash(pair_name)
        filename, _ = get_hashed_file_path(pair_name, file_hash,
                                           kwargs.get('output_dir'))
        align = self._load_or_concatenate(pair, filename, kwargs)

        B, N = align.matrix.shape
        MSA = np.ascontiguousarray(align.matrix.T, dtype=np.int8)      # (N, B)
        # Weights the store already has are used as they are. Otherwise W is
        # None and the pair is reweighted on the GPU it is scored on (run()),
        # which also takes them off these host processes' cores.
        W, Beff = None, None
        if getattr(align, 'weights', None) is not None and align.Beff is not None:
            W, Beff = np.asarray(align.weights, dtype=np.float64), float(align.Beff)
        params = {'name': pair_name, 'pair_id': pair.get('id'), 'B': B, 'N': N,
                'q': 21, 'Beff': Beff, 'file_hash': file_hash,
                'saved_columns': list(align.saved_columns),
                'lengthes': list(align.lengthes)}
        return MSA, W, params

    def collect_score(self, score, params):
        self._keep_gpu_weights(params)
        s16 = score.astype(np.float16)
        analyzer = ComplexAnalyzer(score, saved_columns=params['saved_columns'],
                                       chainA_length=params['lengthes'][0],
                                       EC_threshold=0)
        calibrated = self.Calibrate(analyzer, params["Beff"])
        return {
            'id': params['pair_id'],
            'effnumber': params['Beff'],
            'length': int(params['N']),
            'number': int(params['B']),
            'file_hash': params['file_hash'],
            **calibrated,
            **self._score_fields(s16, params),
        }



# ---------------------------------------------------------------------------
# GPU PLM executor
# ---------------------------------------------------------------------------
class PLMCouplingExecutorGPU(SessionGPUExecutor, BuiltinPLMCalibrator):

    GPU_LABEL = 'plmDCA GPU'

    def __init__(self, config):
        super().__init__(config)
        coupling_cfg = config.get('couple')
        self.fast_approximation = bool(coupling_cfg.get('fast_approximation', False))
        self.col_gap_threshold = float(
            (config.get('align') or {}).get('col_gap_threshold', 0.5))
        # Upper bound on the sites a pair optimises in lockstep (plm_cuda.slots).
        self.site_batch = int(coupling_cfg.get('site_batch', 24))
        # The GPU stops a site on ‖∇f‖∞ ≤ eps_g. eps_conv is NLopt's f/x
        # tolerance on the CPU path, a different rule, so it is not reused.
        self.eps_g = float(coupling_cfg.get('eps_g', 1e-3))
        # fp16 L-BFGS history: 1.12x, an approximation (rounds the curvature pairs)
        self.fp16_history = bool(coupling_cfg.get('fp16_history', False))
        # One Session per GPU, whatever couple.gpu_sessions (GaussDCA's) says:
        # one pair already saturates the GPU, so more Sessions only split its
        # L2 and bandwidth between them. Measured on an H100 over a
        # representative helpy mix: 4% slower with 2, 16% with 4.
        self.gpu_sessions = 1

    def _gpu_module(self):
        from massivora.cpp_bindings import plm_cuda
        return plm_cuda

    def _pair_bytes(self, gpu, params, budget):
        # Fewer sites in lockstep take less memory: halve the batch until the
        # pair fits the largest GPU budget, if it can.
        B, N, q = params['B'], params['N'], params['q']
        batch = self.site_batch
        need = gpu.pair_bytes(B, N, q, batch)
        while need > budget and batch > 1:
            batch //= 2
            need = gpu.pair_bytes(B, N, q, batch)
        params['site_batch'] = batch
        return need

    def _run_on_session(self, sess, MSA, W, params, kwargs):
        if W is None:
            # Reweight on the GPU the pair was admitted to, over every column as
            # Reweight_Sequence does. The counts take less device memory than
            # the pair and are done before it starts, so the reservation covers
            # both.
            m = sess.neighbour_counts(MSA, kwargs['reweighting_threshold'] * MSA.shape[1])
            weights = 1.0 / m.astype(np.int64)      # exactly as Reweight_Sequence
            params['Beff'] = float(weights.sum())
            params['weights'] = weights
            W = weights / params['Beff']
        columns = params.get('columns')
        if columns is not None:
            MSA = np.ascontiguousarray(MSA[:, columns])
        score = sess.score(MSA, W, params['q'], lambdaH=self.lambda_h, lambdaJ=self.lambda_J,
                           maxeval=self.maxit, site_batch=params['site_batch'], eps_g=self.eps_g,
                           fp16_history=self.fp16_history)
        params['stats'] = getattr(sess, 'last_stats', None)     # logged by collect_score
        return score

    @staticmethod
    def _kept_columns(align):
        """The columns Drop_Masked_Gap_Columns keeps, without dropping them."""
        probe = BinaryAlignment()
        probe.matrix = np.arange(align.matrix.shape[1], dtype=np.int32)[None, :]
        probe.saved_columns, probe.lengthes = align.saved_columns, align.lengthes
        probe.Drop_Masked_Gap_Columns()
        return probe.matrix[0].tolist()

    def load_pair_arrays(self, pair, **kwargs):
        pair_name = pair.get('name')
        file_hash = pair.get('file_hash') or compute_file_hash(pair_name)
        filename, _ = get_hashed_file_path(pair_name, file_hash, kwargs.get('output_dir'))
        align = self._load_or_concatenate(pair, filename, kwargs)
        q = int(align.matrix.max()) + 1

        # Weights the store already has are used as they are. Otherwise W is
        # None and the pair is reweighted on the GPU it runs on, which also
        # keeps these host processes off the GPU.
        W, Beff = None, None
        if getattr(align, 'weights', None) is not None and align.Beff is not None:
            Beff = float(align.Beff)
            W = np.asarray(align.weights, dtype=np.float64) / Beff

        # Optimise only the columns the gap mask keeps; the weights stay those
        # of the whole alignment. Without weights yet the whole alignment goes
        # to the GPU, which reweights it and then drops the columns.
        columns = None
        if self.fast_approximation:
            align.Mask_Gap_Columns(self.col_gap_threshold)
            if os.path.exists(filename):
                align.To_Zarr(filename, overwrite=True)
            if W is not None:
                align.Drop_Masked_Gap_Columns()
            else:
                columns = self._kept_columns(align)

        MSA = np.ascontiguousarray(align.matrix, dtype=np.int8)        # (B, N)
        B = MSA.shape[0]
        N = MSA.shape[1] if columns is None else len(columns)
        params = {'name': pair_name, 'pair_id': pair.get('id'), 'B': B, 'N': N, 'q': q,
                  'Beff': Beff, 'file_hash': file_hash, 'columns': columns,
                  'saved_columns': list(align.saved_columns),
                  'lengthes': list(align.lengthes)}
        return MSA, W, params

    def collect_score(self, score, params):
        stats = params.get('stats')
        if stats:
            logging.debug(f"plmDCA GPU pair {params['pair_id']}: {stats['converged']} converged, "
                          f"{stats['stalled']} stalled, {stats['max_iter']} max_iter, "
                          f"{stats['ls_failed']} ls_failed")
        self._keep_gpu_weights(params)
        score = np.asarray(score).astype(np.float16)
        if self.fast_approximation:
            # Expand the score of the kept columns to the whole alignment
            score = self._expand_score(score, params['saved_columns'], params['lengthes'])
        analyzer = ComplexAnalyzer(score, saved_columns=params['saved_columns'],
                                   chainA_length=params['lengthes'][0],
                                   EC_threshold=0)
        calibrated = self.Calibrate(analyzer, params["Beff"])
        return {
            'id': params['pair_id'],
            'effnumber': params['Beff'],
            'length': int(score.shape[0]),
            'number': int(params['B']),
            'file_hash': params['file_hash'],
            **calibrated,
            **self._score_fields(score, params),
        }


EXECUTORS = {
    'plm': (PLMCouplingExecutor, PLMCouplingExecutorGPU),
    'gauss': (GaussCouplingExecutor, GaussCouplingExecutorGPU),
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
    else:
        ExecutorClass = cpu_cls
    logging.info(f"Using executor: {ExecutorClass.__name__}")

    executor = ExecutorClass(cfg)
    executor.run()

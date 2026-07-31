import glob
import logging
import multiprocessing as mp
import os
import subprocess
from itertools import combinations
import pandas as pd

from Bio import ExPASy
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn)

from massivora.config import (_normalize_config_path,
                              load_project_and_system_config)
from massivora.db import (STATUS, connect_db, get_table_names, get_db_path,
                          quote_identifier)
from massivora.utils import setup_logging


class BaseJobLoader(object):
    def __init__(self, project_cfg):
        self.config_path = _normalize_config_path(project_cfg)
        self.config = load_project_and_system_config(self.config_path)
        self.project_path = self.config.get('project').get('project_path')
        self.db_path = get_db_path(self.config)
        self.conda_prefix = os.environ.get('CONDA_PREFIX')

        # Extract table names
        self.table_names = get_table_names(self.config)

        setup_logging(self.config)

    def download_protein(self, protein):
        """
        Download the protein sequence from ExPASy and save it in a text file.

        Parameters
        ----------
        `protein` — str
            The protein swiss-prot ID.
        """
        try:
            output_folder = os.path.join(self.project_path, 'msa', protein)
            os.makedirs(output_folder, 0o755)
        except OSError:
            logging.warning(f"Directory for protein \"{protein}\" already exists")

        align_table = quote_identifier(self.table_names['alignments'])

        conn = connect_db(self.config)
        cursor = conn.cursor()
        try:
            handle = ExPASy.get_sprot_raw(protein)
            with open(os.path.join(output_folder, f"{protein}.txt"), 'w') as f:
                f.write(handle.read())
            cursor.execute(
                f"UPDATE {align_table} SET status = ? WHERE pid = ?",
                (STATUS['NOOPT'], protein),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Failed to fetch protein {protein}: {e}")
            cursor.execute(
                f"UPDATE {align_table} SET status = ? WHERE pid = ?",
                (STATUS['FAILED'], protein),
            )
            conn.commit()
            conn.close()


    def run_download(self, protein_list, processes=4):
        if not protein_list:
            return

        with mp.Pool(processes) as pool:
            with Progress(TextColumn("Downloading proteins..."),
                          BarColumn(),
                          MofNCompleteColumn(),
                          TextColumn("e.t."),
                          TimeElapsedColumn(),
                          refresh_per_second=1,
                    ) as progress:
                task = progress.add_task("Downloading proteins...", total=len(protein_list))
                for _ in pool.imap_unordered(self.download_protein, protein_list):
                    progress.advance(task, 1)
                pool.close()
                pool.join()


class LocalJobLoader(BaseJobLoader):
    def __init__(self, project_cfg):
        super().__init__(project_cfg)

    def _run_detached(self, cmd, log_prefix):
        logging_cfg = self.config.get('logging') or {}
        log_dir = logging_cfg.get('dir')
        if not log_dir:
            log_dir = os.path.join(self.project_path, 'logs')
        elif not os.path.isabs(log_dir):
            log_dir = os.path.join(self.project_path, log_dir)
        os.makedirs(log_dir, mode=0o755, exist_ok=True)

        stdout_path = os.path.join(log_dir, f"{log_prefix}.out")
        stderr_path = os.path.join(log_dir, f"{log_prefix}.err")

        # Run in a new session and detach from the controlling terminal.
        with open(stdout_path, 'ab') as stdout, open(stderr_path, 'ab') as stderr:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )

        logging.info(
            "Started background job %s (pid=%s). Logs: %s",
            log_prefix,
            proc.pid,
            log_dir,
        )
        return proc.pid

    def run_align(self):
        cmd = ['massiworker', 'align', '--config', self.config_path]
        return self._run_detached(cmd, 'align')

    def run_concatenate(self):
        align_table = quote_identifier(self.table_names['alignments'])

        conn = connect_db(self.config)
        cursor = conn.cursor()
        cursor.execute(f"SELECT pid from {align_table} where status = ?", (STATUS['DONE'],))
        done_alignments = [pid for (pid,) in cursor]
        jobs = list(combinations(done_alignments, 2))
        conn.close()

        jobs_file = os.path.join(self.project_path, "concat_jobs.txt")
        with open(jobs_file, 'w') as f:
            for job in jobs:
                f.write(f"{','.join(job)}\n")

        cmd = ['massiworker', 'concat', '--config', self.config_path, jobs_file]
        return self._run_detached(cmd, 'concat')

    def run_couple(self):
        cmd = ['massiworker', 'couple', '--config', self.config_path]
        return self._run_detached(cmd, 'couple')


class SlurmJobLoader(BaseJobLoader):
    """SLURM mode job loader.

    Responsible for generating sbatch scripts, submitting jobs, and recording job ids into job_db.
    Status transitions are handled by massiveilance.py.
    """

    def __init__(self, project_cfg):
        super().__init__(project_cfg)

        self.log_dir = self.config.get('logging').get('dir')
        self.script_dir = os.path.join(self.project_path, self.config.get('batch').get('script_dir'))
        self.job_name_prefix = self.config.get('project').get('name', 'massivora')
        self.time_limit = self.config.get('batch').get('time_limit')
        self.cpu_count = self.config.get('batch').get('cpu_count', 1)
        self.modules = self.config.get('batch').get('modules', [])

        # Cluster-specific #SBATCH directives for routing the coupling step to
        # GPU nodes (empty by default -> ordinary CPU submission).
        self.sbatch_directives = self.config.get('batch').get('sbatch_directives') or []

        os.makedirs(self.script_dir, mode=0o755, exist_ok=True)

    def sbatch(self, script_path):
        rv = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
        if rv.returncode != 0:
            raise RuntimeError(f"sbatch failed: {rv.stderr.strip()}")

        # Typical: "Submitted batch job 123456"
        out = (rv.stdout or '').strip()
        parts = out.split()
        job_id = parts[-1] if parts else None
        if not job_id or not job_id.isdigit():
            raise RuntimeError(f"Unable to parse job id from sbatch output: {out}")
        return job_id

    def write_script(self, script_path, content):
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def default_sbatch_header(self, job_name, cpus_per_task=None, time_limit=None, log_dir="./logs", modules=None, extra_directives=[]):
        if cpus_per_task is None:
            cpus_per_task = 1

        header = (
            "#!/bin/bash\n"
            f"#SBATCH -o {log_dir}/out.%j\n"
            f"#SBATCH -e {log_dir}/err.%j\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks-per-node=1\n"
            f"#SBATCH --cpus-per-task={cpus_per_task}\n"
            f"#SBATCH -J {job_name}\n"
        )
        if time_limit:
            header += f"#SBATCH --time={time_limit}\n"
        # Cluster-specific directives (e.g. GPU partition / --gres) injected as-is.
        for directive in extra_directives:
            directive = str(directive).strip()
            if not directive:
                continue
            if not directive.startswith("#SBATCH"):
                directive = f"#SBATCH {directive}"
            header += f"{directive}\n"
        header += "source /etc/profile\n"
        if modules:
            header += "module purge\n"
            for module in modules:
                header += f"module load {str(module).strip()}\n"
        header += "export OMP_NUM_THREADS=1\n"
        return header

    def run_align(self):
        batch = self.config.get('batch')
        align_cfg = self.config.get('align') or {}
        align_table = quote_identifier(self.table_names['alignments'])

        maximum_nodes = max(1, int(batch.get('maximum_nodes', 1)))
        total_cpu = max(1, int(batch.get('cpu_count', 1)))
        per_job_cpu = max(1, int(align_cfg.get('per_job_cpu', 4)))

        # Each submitted node can claim this many alignment jobs concurrently.
        jobs_per_node = max(1, total_cpu // per_job_cpu)

        conn = connect_db(self.config)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {align_table}")
        total_jobs = int(cursor.fetchone()[0])
        conn.close()

        if total_jobs <= 0:
            logging.error("No alignment jobs found in alignments table; skipping SLURM submission.")
            return

        # Minimal nodes needed to cover all jobs in the first batch, capped by configured maximum.
        nodes_needed = (total_jobs + jobs_per_node - 1) // jobs_per_node
        maximum_nodes = min(maximum_nodes, nodes_needed)

        for node in range(maximum_nodes):
            job_name = f"{self.job_name_prefix}_ali{node}"
            header = self.default_sbatch_header(job_name, cpus_per_task=total_cpu, time_limit=self.time_limit, log_dir=self.log_dir, modules=self.modules)
            cmd = f"conda run -p {self.conda_prefix} massiworker align --config {self.config_path}\n"

            script_path = os.path.join(self.script_dir, f"align_node_{node}.sh")
            self.write_script(script_path, header + cmd)
            job_id = self.sbatch(script_path)

            conn = connect_db(self.config)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (job_id, status) VALUES (?, ?)",
                (str(job_id), 'PENDING')
            )
            conn.commit()
            conn.close()

    def run_concatenate(self):
        align_table = quote_identifier(self.table_names['alignments'])

        conn = connect_db(self.config)
        cursor = conn.cursor()
        cursor.execute(f"SELECT pid from {align_table} where status = ?", (STATUS['DONE'],))
        done_alignments = [pid for (pid,) in cursor]
        jobs = list(combinations(done_alignments, 2))
        conn.close()

        batch = self.config.get('batch')
        concatenate = self.config.get('concatenate')
        maximum_nodes = int(batch.get('maximum_nodes', 1))
        portion = float(concatenate.get('portion', 1.0))
        portion_start = float(concatenate.get('portion_start', 0.0))

        total_concats = len(jobs)
        total_concats_this_slurm = int(total_concats * portion)
        starting_job_index = int(total_concats * portion_start)
        jobs_per_node = total_concats_this_slurm // maximum_nodes + 1

        for node in range(maximum_nodes):
            if starting_job_index >= total_concats_this_slurm + starting_job_index:
                break

            this_node_start = starting_job_index + node * jobs_per_node
            this_node_end = min(
                this_node_start + jobs_per_node,
                total_concats_this_slurm + starting_job_index,
            )
            node_jobs = jobs[this_node_start:this_node_end]
            if not node_jobs:
                continue

            this_node_jobs_file = os.path.join(self.script_dir, f"concat_node_{node}.txt")
            with open(this_node_jobs_file, 'w') as f:
                for job in node_jobs:
                    f.write(f"{','.join(job)}\n")

            job_name = f"{self.job_name_prefix}_cc{node}"
            header = self.default_sbatch_header(job_name, cpus_per_task=self.cpu_count, time_limit=self.time_limit, log_dir=self.log_dir, modules=self.modules)
            cmd = f"conda run -p {self.conda_prefix} massiworker concat --config {self.config_path} {this_node_jobs_file}\n"

            script_path = os.path.join(self.script_dir, f"concat_node_{node}.sh")
            self.write_script(script_path, header + cmd)

            job_id = self.sbatch(script_path)

            conn = connect_db(self.config)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (job_id, status) VALUES (?, ?)",
                (str(job_id), 'PENDING')
            )
            conn.commit()
            conn.close()

    def run_couple(self):
        batch = self.config.get('batch')
        maximum_nodes = int(batch.get('maximum_nodes', 1))

        for node in range(maximum_nodes):
            job_name = f"{self.job_name_prefix}_cp{node}"
            header = self.default_sbatch_header(job_name, cpus_per_task=self.cpu_count, time_limit=self.time_limit, log_dir=self.log_dir, modules=self.modules, extra_directives=self.sbatch_directives)
            cmd = f"conda run -p {self.conda_prefix} massiworker couple --config {self.config_path}"

            script_path = os.path.join(self.script_dir, f"coupling_node_{node}.sh")
            self.write_script(script_path, header + cmd)

            job_id = self.sbatch(script_path)

            conn = connect_db(self.config)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (job_id, status) VALUES (?, ?)",
                (str(job_id), 'PENDING')
            )
            conn.commit()
            conn.close()

    # Map a monitored pipeline stage to the sbatch scripts it generates.
    _STAGE_SCRIPT_PREFIX = {'align': 'align_node_', 'couple': 'coupling_node_'}

    def resubmit(self, stage):
        """
        Re-submit the sbatch scripts already generated for ``stage``. Returns the number of scripts resubmitted.
        """
        prefix = self._STAGE_SCRIPT_PREFIX.get(stage)
        if prefix is None:
            raise ValueError(f"Cannot resubmit unknown stage: {stage!r}")

        scripts = sorted(glob.glob(os.path.join(self.script_dir, f"{prefix}*.sh")))

        submitted = 0
        for script_path in scripts:
            job_id = self.sbatch(script_path)

            conn = connect_db(self.config)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (job_id, status) VALUES (?, ?)",
                (str(job_id), 'PENDING')
            )
            conn.commit()
            conn.close()
            submitted += 1
        return submitted

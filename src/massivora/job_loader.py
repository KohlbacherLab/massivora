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
from massivora.db import STATUS, connect_db
from massivora.logging_utils import setup_logging


class BaseJobLoader(object):
    def __init__(self, project_cfg):
        self.config_path = _normalize_config_path(project_cfg)
        self.config = load_project_and_system_config(self.config_path)
        paths = self.config.get('paths')
        self.project_path = self.config.get('project').get('project_path')
        db_rel = paths.get('job_db')
        self.db_path = os.path.join(self.project_path, db_rel) if self.project_path and db_rel else None
        self.conda_env = self.config.get('manager').get('conda_env')
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
        conn = connect_db(self.db_path)
        cursor = conn.cursor()
        try:
            handle = ExPASy.get_sprot_raw(protein)
            with open(os.path.join(output_folder, f"{protein}.txt"), 'w') as f:
                f.write(handle.read())
            cursor.execute(
                "INSERT OR IGNORE INTO alignments (pid, status) VALUES (?, ?)",
                (protein, -1),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Failed to fetch protein {protein}: {e}")
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

    def run_align(self, protein_list):
        cmd = ['massiworker', 'align', '--config', self.config_path, self.config.get('project').get('proteins_list')]
        return subprocess.run(cmd).returncode

    def run_concatenate(self):
        cmd = ['massiworker', 'concat', '--config', self.config_path]
        return subprocess.run(cmd).returncode

    def run_couple(self):
        cmd = ['massiworker', 'couple', '--config', self.config_path]
        return subprocess.run(cmd).returncode


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

    def default_sbatch_header(self, job_name, cpus_per_task=None, time_limit=None, log_dir="./logs", modules=None):
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
        if modules:
            header += "module purge\n"
            for module in modules:
                header += f"module load {module}\n"
        header += "export OMP_NUM_THREADS=1\n"
        return header

    def run_align(self, protein_list):
        batch = self.config.get('batch')
        align = self.config.get('align')
        maximum_nodes = int(batch.get('maximum_nodes', 1))
        portion = float(align.get('portion', 1.0))
        portion_start = float(align.get('portion_start', 0.0))

        total_proteins = len(protein_list)
        total_proteins_this_slurm = int(total_proteins * portion)
        starting_protein_index = int(total_proteins * portion_start)
        proteins_per_node = total_proteins_this_slurm // maximum_nodes + 1

        for node in range(maximum_nodes):
            if starting_protein_index >= total_proteins_this_slurm + starting_protein_index:
                break

            this_node_start = starting_protein_index + node * proteins_per_node
            this_node_end = min(
                this_node_start + proteins_per_node,
                total_proteins_this_slurm + starting_protein_index,
            )
            proteins = protein_list[this_node_start:this_node_end]
            if not proteins:
                continue
            
            this_node_proteins_file = os.path.join(self.script_dir, f"align_node_{node}.txt")
            with open(this_node_proteins_file, 'w') as f:
                for pid in proteins:
                    f.write(f"{pid}\n")

            job_name = f"{self.job_name_prefix}_ali{node}"
            header = self.default_sbatch_header(job_name, cpus_per_task=self.cpu_count, time_limit=self.time_limit, log_dir=self.log_dir, modules=self.modules)
            cmd = f"conda run -n {self.conda_env} massiworker align --config {self.config_path} {this_node_proteins_file}\n"

            script_path = os.path.join(self.script_dir, f"align_node_{node}.sh")
            self.write_script(script_path, header + cmd)

            job_id = self.sbatch(script_path)

            conn = connect_db(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (job_id) VALUES (?)",
                (str(job_id),)
            )

            # Record job_id for each protein
            for pid in proteins:
                cursor.execute(
                    "UPDATE alignments SET job_id = ?, status = ? WHERE pid = ?",
                    (str(job_id), STATUS['PENDING'], pid),
                )
            conn.commit()
            conn.close()

    def run_concatenate(self):
        project_path = (self.config.get('project') or {}).get('project_path')
        if not project_path:
            raise SystemExit('Missing config: project.project_path')

        conn = connect_db(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT pid from alignments where status = ?", (STATUS['DONE'],))
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
            cmd = f"conda run -n {self.conda_env} massiworker concat --config {self.config_path} {this_node_jobs_file}\n"

            script_path = os.path.join(self.script_dir, f"concat_node_{node}.sh")
            self.write_script(script_path, header + cmd)

            job_id = self.sbatch(script_path)

            conn = connect_db(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (job_id) VALUES (?)",
                (str(job_id),)
            )
            conn.commit()
            conn.close()

    def run_couple(self):
        batch = self.config.get('batch')
        maximum_nodes = int(batch.get('maximum_nodes', 1))
        coupling_cfg = self.config.get('coupling')
        portion = float(coupling_cfg.get('portion', 1.0))
        portion_start = float(coupling_cfg.get('portion_start', 0.0))

        # Record job_id for coupling task id range
        conn = connect_db(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM couplings WHERE status != ?",
            (STATUS['DONE'],),
        )
        total_proteins = cursor.fetchone()[0]
        conn.close()

        total_couplings = int(total_proteins * (total_proteins - 1) / 2)
        total_couplings_this_slurm = int(total_couplings * portion) + 1
        starting_coupling_index = int(total_couplings * portion_start) + 1
        per_node = total_couplings_this_slurm // maximum_nodes + 1

        for node in range(maximum_nodes):
            if starting_coupling_index >= total_couplings_this_slurm + starting_coupling_index:
                break

            job_name = f"{self.job_name_prefix}_cp{node}"
            header = self.default_sbatch_header(job_name, cpus_per_task=self.cpu_count, time_limit=self.time_limit, log_dir=self.log_dir, modules=self.modules)
            cmd = f"conda run -n {self.conda_env} massiworker couple --config {self.config_path}"


            script_path = os.path.join(self.script_dir, f"coupling_node_{node}.sh")
            self.write_script(script_path, header + cmd)

            job_id = self.sbatch(script_path)

            conn = connect_db(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (job_id) VALUES (?)",
                (str(job_id),)
            )
            conn.commit()
            conn.close()

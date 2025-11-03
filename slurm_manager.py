import argparse
import glob
import logging
import multiprocessing as mp
import os
import sqlite3
from functools import partial

import pandas as pd
import psutil
from alignment import ZarrAlignment
from Bio import ExPASy
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn)

# Set up logging
logging.basicConfig(
    filename='errors.log',
    level=logging.WARNING,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def download_protein(protein):
    """
    Download the protein sequence from ExPASy and save it in a text file.

    Parameters
    ----------
    `protein` — str
        The protein swiss-prot ID.
    """
    # should be extracted from config file
    project_path = '/ptmp/sunh/helpy'
    try:
        output_folder = os.path.join(project_path, 'msa', protein)
        os.makedirs(output_folder, 0o755)
    except OSError:
        logging.warning(f"Directory for protein \"{protein}\" already exists")
    try:
        handle = ExPASy.get_sprot_raw(protein)
        with open(os.path.join(output_folder, f"{protein}.txt"), 'w') as f:
            f.write(handle.read())
    except Exception as e:
        logging.error(f"Failed to fetch protein {protein}: {e}")
        return

def concatenate_index(alignments, alignment_dir, output_dir):
    """
    Concatenate all alignments in the given list with all alignments that come after it.
    Save the concatenated alignments in the output directory.

    Parameters
    ----------
    `alignments` — int
        The alignments list to concatenate.
    `alignment_dir` — str
        The directory where the alignments are stored.
    `output_dir` — str
        The directory where the concatenated alignments will be saved.
    """
    align1 = ZarrAlignment(os.path.join(alignment_dir, alignments[0]))
    rlist = []
    for align2_name in alignments[1:]:
        align2 = ZarrAlignment(os.path.join(alignment_dir, align2_name))
        concatenated_align = align1 + align2
        filename = os.path.join(output_dir, f"{alignments[0]}-{align2_name}.a2m")
        rlist.append((alignments[0], align2_name, concatenated_align.shape[1], concatenated_align.shape[0], False))
        concatenated_align.To_Text(False, filename)
    return rlist

def cut_alignment_list(alignments):
    for i in range(len(alignments) - 1):
        yield alignments[i:]

def create_job_DB(db_name="job_db.sqlite3"):
    # TODO: Get path from config file
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        pid TEXT NOT NULL, 
        start INTEGER NOT NULL, 
        end INTEGER NOT NULL, 
        align_nseqs INTEGER, 
        filtered_nseqs INTEGER,
        finished BOOLEAN NOT NULL
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS couplings (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        pid1 TEXT NOT NULL, 
        pid2 TEXT NOT NULL, 
        length INTEGER NOT NULL, 
        number INTEGER NOT NULL, 
        finished BOOLEAN NOT NULL,
        UNIQUE (pid1, pid2)
    );
    ''')

    conn.commit()
    conn.close()

def mark_finished_jobs(csv_filenames, db_name="job_db.sqlite3"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    if isinstance(csv_filenames, str):
        pid1, pid2 = os.path.splitext(os.path.basename(csv_filenames))[0].split('-')
        cursor.execute('''UPDATE couplings SET finished = True WHERE pid1 = ? AND pid2 = ?''', (pid1, pid2))
    else:
        pidpairs = []
        for csv_filename in csv_filenames:
            pidpairs.append(os.path.splitext(os.path.basename(csv_filename))[0].split('-'))
        cursor.executemany('''UPDATE couplings SET finished = True WHERE pid1 = ? AND pid2 = ?''', pidpairs)
    conn.commit()
    conn.close()

def generate_batch_script(node_id, job_cmd, script_dir, job_name):
    if not os.path.exists(script_dir):
        os.makedirs(script_dir, mode=0o764)
    with open(os.path.join(script_dir, f'node_{node_id}.sh'), 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH -o ./out.%j\n")
        f.write("#SBATCH -e ./err.%j\n")
        f.write("#SBATCH --nodes=1\n")
        f.write(f"#SBATCH --cpus-per-task={psutil.cpu_count(logical=False)}\n")
        f.write("#SBATCH --ntasks-per-node=1\n")
        f.write(f"#SBATCH -J {job_name}_{node_id}\n")
        f.write(f"#SBATCH --time=24:00:00\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write("module purge\n")
        # f.write("module load gcc/13 impi/2021.11\n")
        f.write("module load anaconda/3/2023.03\n")
        f.write("module load julia/1.11\n")
        f.write(f"{job_cmd}\n")
    return os.path.join(script_dir, f'node_{node_id}.sh')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein alignment and download script')
    parser.add_argument('-d', '--download', action='store_true', help='Download protein mode')
    parser.add_argument('-a', '--align', action='store_true', help='Align protein mode')
    parser.add_argument('-c', '--concatenate', action='store_true', help='Concatenate protein mode')
    parser.add_argument('-p', '--coupling', action='store_true', help='Coupling calculation mode')

    args = parser.parse_args()

    # TODO: Get path from config file
    alignment_dir = '/ptmp/sunh/helpy/msa'
    output_dir = '/ptmp/sunh/helpy/ecs'
    script_dir = "/ptmp/sunh/helpy/jobs"

    # Read from config or auto-detect
    MAXIMUM_NODES = 300
    PORTION = 0.4
    STARTING_PORTION = 0.4

    # TODO: Get path from config file
    job_DB_dir = '/ptmp/sunh/helpy/job_db.sqlite3'
    create_job_DB(job_DB_dir)

    # TODO: Get path from user input or config file
    protein_list = pd.read_csv("/ptmp/sunh/helpy/data/helpy_proteome_2align.csv")['uid'].tolist()
    TOTAL_PROTEINS = len(protein_list)

    # download proteins on login node
    # computing nodes have no internet access
    if args.download:
        with mp.Pool(4) as pool:
            with Progress(TextColumn("Downloading proteins..."),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("e.t."),
                    TimeElapsedColumn(),
                    refresh_per_second=1) as progress:
                task = progress.add_task("Downloading proteins...", total=len(protein_list))
                for _ in pool.imap_unordered(download_protein, protein_list):
                    progress.advance(task, 1)
                pool.close()
                pool.join()

    # homologue search
    if args.align:
        TOTAL_PROTEINS_THIS_SLURM = int(TOTAL_PROTEINS * PORTION)
        STARTING_PROTEIN_INDEX = int(TOTAL_PROTEINS * STARTING_PORTION)
        PROTEINS_PER_NODE = TOTAL_PROTEINS_THIS_SLURM // MAXIMUM_NODES + 1
        for node in range(MAXIMUM_NODES):
            if STARTING_PROTEIN_INDEX >= TOTAL_PROTEINS_THIS_SLURM + STARTING_PROTEIN_INDEX:
                break
            this_node_start = STARTING_PROTEIN_INDEX + node * PROTEINS_PER_NODE
            this_node_end = min(this_node_start + PROTEINS_PER_NODE, TOTAL_PROTEINS_THIS_SLURM + STARTING_PROTEIN_INDEX)
            # TODO: Get path from env variable or config file
            job_cmd = f"/u/sunh/conda-envs/ecrun/bin/python /u/sunh/EVsnap/executors/align.py -a {' '.join(protein_list[this_node_start:this_node_end])}"
            job_name = "helpyali"
            script_path = generate_batch_script(node, job_cmd, script_dir, job_name)
            os.system(f"sbatch {script_path}")


    # concatenating the alignments
    if args.concatenate:
        # TODO: Get path from env variable or config file
        job_cmd = f"/u/sunh/conda-envs/ecrun/bin/python /u/sunh/EVsnap/executors/concatenate.py -i /ptmp/sunh/helpy/msa -o /ptmp/sunh/helpy/ecs"
        job_name = "helpycc"
        script_path = generate_batch_script(0, job_cmd, script_dir, job_name)
        os.system(f"sbatch {script_path}")

    # perform coupling analysis
    if args.coupling:
        # mark finished jobs before the next start
        # csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
        # mark_finished_jobs(csv_files, db_name=job_DB_dir)

        TOTAL_COUPLINGS = int(TOTAL_PROTEINS * (TOTAL_PROTEINS - 1) / 2)
        TOTAL_COUPLINGS_THIS_SLURM = int(TOTAL_COUPLINGS * PORTION) + 1
        STARTING_COUPLING_INDEX = int(TOTAL_COUPLINGS * STARTING_PORTION) + 1
        PROTEINS_PER_NODE = TOTAL_COUPLINGS_THIS_SLURM // MAXIMUM_NODES + 1
        for node in range(MAXIMUM_NODES):
            if STARTING_COUPLING_INDEX >= TOTAL_COUPLINGS_THIS_SLURM + STARTING_COUPLING_INDEX:
                break
            this_node_start = STARTING_COUPLING_INDEX + node * PROTEINS_PER_NODE
            this_node_end = min(this_node_start + PROTEINS_PER_NODE, TOTAL_COUPLINGS_THIS_SLURM + STARTING_COUPLING_INDEX)
            # TODO: Get path from env variable or config file
            job_cmd = f"/u/sunh/conda-envs/ecrun/bin/python /u/sunh/EVsnap/executors/coupling.py -m plm -b {this_node_start} -e {this_node_end} -lh 0.01 -le 0.01"
            job_name = "helpycp"
            script_path = generate_batch_script(node, job_cmd, script_dir, job_name)
            os.system(f"sbatch {script_path}")
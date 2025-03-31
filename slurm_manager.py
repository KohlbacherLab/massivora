import os
import glob
import sqlite3
from alignment import ZarrAlignment
import multiprocessing as mp
from functools import partial
import psutil
import numpy as np

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
        concatenated_align.Downsample_Randomly(to=5000)
        filename = os.path.join(output_dir, f"{alignments[0]}-{align2_name}.npy")
        rlist.append((alignments[0], align2_name, concatenated_align.shape[1], concatenated_align.shape[0], False))
        concatenated_align.To_Npy(filename)
    return rlist

def cut_alignment_list(alignments):
    for i in range(len(alignments) - 1):
        yield alignments[i:]

def create_job_DB(db_name="job_db.sqlite3"):
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
        f.write(f"#SBATCH -J {job_name}_node_{node_id}\n")
        f.write(f"#SBATCH --time=24:00:00\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write("module purge\n")
        f.write("module load gcc/13 impi/2021.11\n")
        f.write("module load anaconda/3/2023.03\n")
        f.write("module load julia/1.11\n")
        f.write(f"{job_cmd}\n")
    return os.path.join(script_dir, f'node_{node_id}.sh')


MAXIMUM_NODES = 300
TOTAL_PROTEINS = 1554
PORTION = 0.3
STARTING_PORTION = 0.4

if __name__ == '__main__':
    alignment_dir = '/ptmp/sunh/helpy/msa'
    output_dir = '/ptmp/sunh/helpy/ecs'
    script_dir = "/ptmp/sunh/helpy/jobs"
    
    job_DB_dir = '/ptmp/sunh/helpy/job_db.sqlite3'
    create_job_DB(job_DB_dir)

    # concatenating the alignments
    # alignments = [os.path.basename(d) for d in os.listdir(alignment_dir) if os.path.isdir(os.path.join(alignment_dir, d))]
    # concatenate_partial = partial(concatenate_index, alignment_dir=alignment_dir, output_dir=output_dir)
    # with mp.Pool() as pool:
    #     rv = pool.map(concatenate_partial, cut_alignment_list(alignments))
    # conn = sqlite3.connect("job_db.sqlite3")
    # for r in rv:
    #     conn.executemany('''INSERT INTO couplings (pid1, pid2, length, number, finished) VALUES (?, ?, ?, ?, ?)''', r)
    # conn.commit()
    # conn.close()

    # mark finished jobs before the next start
    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
    mark_finished_jobs(csv_files, db_name=job_DB_dir)

    # perform coupling analysis
    TOTAL_COUPLINGS = int(TOTAL_PROTEINS * (TOTAL_PROTEINS - 1) / 2)
    TOTAL_COUPLINGS_THIS_SLURM = int(TOTAL_COUPLINGS * PORTION) + 1
    STARTING_COUPLING_INDEX = int(TOTAL_COUPLINGS * STARTING_PORTION) + 1
    PROTEINS_PER_NODE = TOTAL_COUPLINGS_THIS_SLURM // MAXIMUM_NODES + 1
    for node in range(MAXIMUM_NODES):
        if STARTING_COUPLING_INDEX >= TOTAL_COUPLINGS_THIS_SLURM + STARTING_COUPLING_INDEX:
            break
        this_node_start = STARTING_COUPLING_INDEX + node * PROTEINS_PER_NODE
        this_node_end = min(this_node_start + PROTEINS_PER_NODE, TOTAL_COUPLINGS_THIS_SLURM + STARTING_COUPLING_INDEX)
        job_cmd = f"/u/sunh/conda-envs/ecrun/bin/python /u/sunh/EVsnap/helpy_coupling.py {this_node_start} {this_node_end}"
        job_name = "helpy"
        script_path = generate_batch_script(node, job_cmd, script_dir, job_name)
        os.system(f"sbatch {script_path}")
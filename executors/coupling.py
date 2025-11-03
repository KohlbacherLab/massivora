import argparse
import logging
import multiprocessing as mp
import os
import sqlite3
import subprocess
import sys
from functools import partial
from shutil import copyfile, move

import psutil
from concatenate import concatenate_alignments

# Set up logging
logging.basicConfig(
    filename='errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def run_plmc(index, alignment_dir, lambda_h=None, lambda_J=None, lambda_g=None):
    """
    Run coupling with the given index in job database.

    Parameters
    ----------
    `index` — int
        The ID of the coupling task.
    `alignment_dir` — str
        The directory where the alignment files are stored, output files will be saved there too.
    """
    threads = '8'
    conn = sqlite3.connect("file:/ptmp/sunh/helpy/job_db.sqlite3?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    row = cursor.execute("SELECT * FROM couplings WHERE id = ?", (index,)).fetchone()
    conn.close()
    if row['finished']:
        return
    pid1, pid2 = row['pid1'], row['pid2']
    concat_filename = os.path.join(alignment_dir, f"{pid1}-{pid2}.a2m")
    # ali = concatenate_alignments(row['pid1'], row['pid2'])
    # ali.To_Text(restore_gaps=False, filename=concat_filename)

    rv = subprocess.run(['/u/sunh/apps/plmc', '-c', os.path.join(alignment_dir, f"{pid1}-{pid2}.txt"), '-o', '/dev/null', '-f', '0', '-g', '-m', '100', '-t', '0.2', '-lh', str(lambda_h), '-le', str(lambda_J*(row['length']-1)*20), '-n', threads, concat_filename])
    if rv.returncode == 0:
        os.remove(concat_filename)
        conn = sqlite3.connect("file:/ptmp/sunh/helpy/job_db.sqlite3?mode=rw", uri=True)
        cursor = conn.cursor()
        cursor.execute("UPDATE couplings SET finished = 1 WHERE id = ?", (index,))
        conn.commit()
        conn.close()

def copy_to(index, dest, alignment_dir):
    conn = sqlite3.connect("file:/ptmp/sunh/helpy/job_db.sqlite3?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    row = cursor.execute("SELECT * FROM couplings WHERE id = ?", (index,)).fetchone()
    conn.close()
    pid1, pid2 = row['pid1'], row['pid2']
    alignment = os.path.join(alignment_dir, f"{pid1}-{pid2}.npy")

    if os.path.exists(alignment):
        copyfile(alignment, dest)
    else:
        logging.error(f"Alignment file {alignment} does not exist.")

def move_to(index, dest, alignment_dir):
    conn = sqlite3.connect("file:/ptmp/sunh/helpy/job_db.sqlite3?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    row = cursor.execute("SELECT * FROM couplings WHERE id = ?", (index,)).fetchone()
    conn.close()
    pid1, pid2 = row['pid1'], row['pid2']
    alignment = os.path.join(alignment_dir, f"{pid1}-{pid2}.npy")
    if os.path.exists(alignment):
        move(alignment, dest)
    else:
        logging.error(f"Alignment file {alignment} does not exist.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein alignment and download script')
    parser.add_argument('-m', '--method', choices=['plm', 'other'], default='plm', help='Coupling calculation method')
    parser.add_argument('-b', '--begin', help='Beginning of the range')
    parser.add_argument('-e', '--end', help='End of the range')
    parser.add_argument('-lh', '--lambda_h', help='Lambda_h value')
    parser.add_argument('-le', '--lambda_J', help='Lambda_J value')
    # parser.add_argument('-lg', '--lambda_g', help='Lambda_g value')

    args = parser.parse_args()

    p_jobs = int(psutil.cpu_count(logical=False)/4)
    if args.method == 'plm':
        coupling_partial = partial(run_plmc, alignment_dir='/ptmp/sunh/helpy/ecs', lambda_h=str(args.lambda_h), lambda_J=str(args.lambda_J))
    with mp.Pool(p_jobs) as pool:
        pool.map(coupling_partial, range(int(args.begin), int(args.end)))

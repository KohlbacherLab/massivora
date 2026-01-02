import argparse
import logging
import multiprocessing as mp
import os
from functools import partial
from itertools import combinations

import psutil

from massivora.config import load_project_and_system_config
from massivora.alignment import ZarrAlignment
from massivora.db import connect_db, STATUS
from massivora.logging_utils import setup_logging


def analyze_input_file(proteins_file):
    with open(proteins_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    results = [tuple(line.strip().split(',')) for line in lines if line.strip()]
    return results

def concatenate(pair):
    pid1, pid2 = pair
    align1 = ZarrAlignment(os.path.join(alignment_dir, pid1))
    align2 = ZarrAlignment(os.path.join(alignment_dir, pid2))
    concatenated_align = align1 + align2
    if downsample:
        concatenated_align.Downsample_Randomly(to=downsample_to)
    concatenated_align.Reweight_Sequence(x=reweighting_threshold)
    filename = os.path.join(output_dir, f"{pid1}-{pid2}")

    try:
        concatenated_align.To_Zarr(filename, overwrite=True)
        concatenated_align.To_Text(False, filename+'.a2m')
    except Exception as e:
        logging.error(f"Failed to save concatenated alignment {pid1}-{pid2}: {e}")
    return (pid1, pid2, concatenated_align.matrix.shape[1], concatenated_align.matrix.shape[0], concatenated_align.Beff, STATUS['NOOPT'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein concatenation script')
    parser.add_argument('--config', required=True, help='Project config YAML')
    parser.add_argument('proteins', type=str, nargs='?', default=None, help='Protein list file')

    args = parser.parse_args()

    cfg = load_project_and_system_config(args.config)
    project_path = cfg.get('project').get('project_path')
    job_db = os.path.join(project_path, cfg.get('paths').get('job_db'))
    if not job_db:
        raise SystemExit('Missing config: paths.job_db')
    alignment_dir = os.path.join(project_path, cfg.get('paths').get('monomers'))
    output_dir = os.path.join(project_path, cfg.get('paths').get('couplings'))
    down_cfg = cfg.get('concatenate').get('downsample')
    downsample = bool(down_cfg.get('enabled', False))
    downsample_to = int(down_cfg.get('to', 10000))
    reweighting_threshold = float(cfg.get('concatenate').get('reweighting_threshold', 0.8))

    setup_logging(cfg)

    logical_cpus = psutil.cpu_count(logical=True)

    if not args.proteins:
        conn = connect_db(job_db)
        cursor = conn.cursor()
        cursor.execute("SELECT pid from alignments where status = ?", (STATUS['DONE'],))
        done_alignments = [pid for (pid,) in cursor]
        pairs = list(combinations(done_alignments, 2))
        conn.close()
    else:
        pairs = analyze_input_file(args.proteins)

    with mp.Pool(logical_cpus) as pool:
        rv = pool.map(concatenate, pairs)
    pool.join()
    pool.close()

    conn = connect_db(job_db)
    conn.executemany(
        '''INSERT OR IGNORE INTO couplings (pid1, pid2, length, number, effnumber, status) VALUES (?, ?, ?, ?, ?, ?)''',
        [(pid1, pid2, length, number, effnumber, status) for (pid1, pid2, length, number, effnumber, status) in rv],
    )
    conn.commit()
    conn.close()

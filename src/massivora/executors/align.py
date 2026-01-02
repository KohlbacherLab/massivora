import os
from functools import partial
import logging
import psutil
import argparse
import multiprocessing as mp
from Bio import SeqIO

from massivora.config import load_project_and_system_config
from massivora.db import connect_db, STATUS
from massivora.logging_utils import setup_logging
from massivora.alignment import TextAlignment


def align_record(protein, **kwargs):
    """
    Create a folder for the protein, perform alignment, filtering, and save as Zarr format.

    Parameters
    ----------
    `protein` — str
        The protein swiss-prot ID.
    """
    output_folder = os.path.join(kwargs.get("project_path"), 'msa', protein)
    if not os.path.exists(output_folder):
        logging.error(f"Directory for protein \"{protein}\" doesn't exist. Please download it first.")
        return

    conn = connect_db(kwargs.get("job_db"))
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM alignments WHERE pid = ?", (protein,))
    row = cursor.fetchone()
    if row and row[0] == STATUS['DONE']:
        logging.info(f"Alignment for protein \"{protein}\" is already done. Skipping.")
        return
    try:
        with open(os.path.join(output_folder, f"{protein}.txt")) as f:
            record = SeqIO.read(f, 'swiss')
        alignment = TextAlignment(record)
        cursor.execute(
            "UPDATE alignments SET start = ?, end = ?, status = ? WHERE pid = ?",
            (1, len(alignment[0]), STATUS['RUNNING'], protein),
        )
        conn.commit()
    except Exception as e:
        logging.error(f"Failed to load protein {protein}: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return

    params = {
        'iterations': kwargs.get("align_iterations"),
        'binary': kwargs.get("align_binary"),
        'working_dir': output_folder,
        'cpu': kwargs.get("align_cpu"),
        'database': kwargs.get("align_database"),
    }

    alignment.AnalogueSearch(threshold=kwargs.get("align_analogue_threshold"), **params)
    align_nseqs = len(alignment)

    alignment.Filtering_MSA_Gap(kwargs.get("row_gap_threshold"), kwargs.get("col_gap_threshold"))
    alignment.Filtering_MSA_Invalid()
    alignment.Best_Reciprocal_Hit(kwargs.get("align_paralogue_threshold"))
    filtered_nseqs = len(alignment)

    alignment.To_Zarr(output_folder)

    try:
        cursor.execute(
            "UPDATE alignments SET align_nseqs = ?, filtered_nseqs = ?, status = 0 WHERE pid = ?",
            (align_nseqs, filtered_nseqs, protein),
        )
        conn.commit()
    except Exception as e:
        logging.error(f"Alignment failed for {protein}: {e}")
        try:
            cursor.execute("UPDATE alignments SET status = 3 WHERE pid = ?", (protein,))
            conn.commit()
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein alignment and download script')
    parser.add_argument('--config', required=True, help='Project config YAML')
    parser.add_argument('proteins', type=str, help='Protein list file')

    args = parser.parse_args()

    with open(args.proteins, 'r', encoding='utf-8') as f:
        protein_list = [line.strip() for line in f.read().splitlines() if line.strip()]

    cfg = load_project_and_system_config(args.config)
    setup_logging(cfg)
    align_cfg = cfg.get('align')
    project_path = cfg.get('project').get('project_path')
    job_db = os.path.join(project_path, cfg.get('paths').get('job_db'))

    logical_cpus = psutil.cpu_count(logical=True)
    partial_align_record = partial(
        align_record,
        project_path=project_path,
        job_db=job_db,
        align_iterations=int(align_cfg.get('iterations', 5)),
        align_binary=align_cfg.get('jackhmmer_binary'),
        align_cpu=int(align_cfg.get('per_job_cpu', 4)),
        align_database=align_cfg.get('database'),
        align_analogue_threshold=align_cfg.get('analogue_threshold', 0.2),
        row_gap_threshold=align_cfg.get('row_gap_threshold', 0.5),
        col_gap_threshold=align_cfg.get('col_gap_threshold', 0.5),
        align_paralogue_threshold=align_cfg.get('paralogue_threshold', 0.9),
    )

    with mp.Pool(max(1, logical_cpus // int(align_cfg.get('per_job_cpu', 4)))) as pool:
        pool.map(partial_align_record, protein_list)

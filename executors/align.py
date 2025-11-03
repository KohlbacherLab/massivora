import os
import sys
import logging
import psutil
import argparse

import pandas as pd
import multiprocessing as mp
from Bio import AlignIO, SeqIO
from Bio import ExPASy
from alignment import TextAlignment
from rich.progress import Progress, TimeElapsedColumn, TextColumn, BarColumn, MofNCompleteColumn

# Set up logging
logging.basicConfig(
    filename='errors.log',
    level=logging.WARNING,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# should be extracted from config file
project_path = '/ptmp/sunh/helpy'

def align_record(protein):
    """
    Create a folder for the protein, perform alignment, filtering, and save as Zarr format.

    Parameters
    ----------
    `protein` — str
        The protein swiss-prot ID.
    """
    try:
        output_folder = os.path.join(project_path, 'msa', protein)
        os.makedirs(output_folder, 0o755)
    except OSError:
        logging.warning(f"Directory for protein \"{protein}\" already exists")
    try:
        with open(os.path.join(output_folder, f"{protein}.txt")) as f:
            record = SeqIO.read(f, 'swiss')
        alignment = TextAlignment(record)
    except Exception as e:
        logging.error(f"Failed to fetch protein {protein}: {e}")
        return
    # Params should be extracted from config file
    params = {
        'iterations': 5,
        'binary': '/u/sunh/apps/hmmer/bin/jackhmmer',
        'working_dir': output_folder,
        'cpu': 4,
        'database': '/u/sunh/uniprot_2025_6.fasta'
    }
    alignment.AnalogueSearch(threshold=0.2, **params)
    alignment.Filtering_MSA_Gap(50, 50)
    alignment.To_Zarr(output_folder)

def download_protein(protein):
    """
    Download the protein sequence from ExPASy and save it in a text file.

    Parameters
    ----------
    `protein` — str
        The protein swiss-prot ID.
    """
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein alignment and download script')
    parser.add_argument('-d', '--download', action='store_true', help='Download protein mode')
    parser.add_argument('-a', '--align', action='store_true', help='Align protein mode')
    parser.add_argument('proteins', nargs='+', help='Protein identifier(s)')

    args = parser.parse_args()

    physical_cpus = psutil.cpu_count(logical=False)

    if args.download:
        logging.info(f"Downloading protein {args.proteins}...")
        with mp.Pool(physical_cpus) as pool:
            pool.map(download_protein, args.proteins)
    elif args.align:
        logging.info(f"Aligning protein {args.proteins}...")
        with mp.Pool(physical_cpus // 2) as pool:
            pool.map(align_record, args.proteins)
    else:
        print("Please specify either -d/--download or -a/--align mode")
        sys.exit(1)
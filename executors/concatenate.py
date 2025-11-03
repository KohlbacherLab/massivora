import logging
import multiprocessing as mp
import os
import sys
sys.path.append('/u/sunh/EVsnap')  # to import alignment module
import argparse
import sqlite3
from alignment import ZarrAlignment
from functools import partial

import psutil

# Set up logging
logging.basicConfig(
    filename='errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def concatenate_alignments(alignment1, alignment2, output_dir=None):
    """
    Concatenate two alignments and save the result in the output directory.

    Parameters
    ----------
    `alignment1` — ZarrAlignment
        The first alignment to concatenate.
    `alignment2` — ZarrAlignment
        The second alignment to concatenate.
    `output_dir` — str
        The directory where the concatenated alignment will be saved.
    """
    concatenated_align = alignment1 + alignment2
    concatenated_align.Downsample_Randomly(to=10000)
    
    if not output_dir:
        output_dir = os.path.abspath(alignment1)

    # filename = os.path.join(output_dir, f"{alignment1}-{alignment2}.npy")
    # concatenated_align.To_Npy(filename)

    return concatenated_align

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
        concatenated_align.Downsample_Randomly(to=10000)
        filename = os.path.join(output_dir, f"{alignments[0]}-{align2_name}.a2m")
        rlist.append((alignments[0], align2_name, concatenated_align.matrix.shape[1], concatenated_align.matrix.shape[0], False))
        try:
            concatenated_align.To_Text(False, filename)
        except Exception as e:
            logging.error(f"Failed to save concatenated alignment {alignments[0]}-{align2_name}: {e}")
    return rlist

def cut_alignment_list(alignments):
    for i in range(len(alignments) - 1):
        yield alignments[i:]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein alignment and download script')
    parser.add_argument('-i', '--input', required=True, help='Input alignments directory')
    parser.add_argument('-o', '--output', required=True, help='Output alignments directory')

    args = parser.parse_args()
    alignment_dir = args.input
    output_dir = args.output

    logical_cpus = psutil.cpu_count(logical=True)

    alignments = [os.path.basename(d) for d in os.listdir(alignment_dir) if os.path.isdir(os.path.join(alignment_dir, d))]
    concatenate_partial = partial(concatenate_index, alignment_dir=alignment_dir, output_dir=output_dir)
    with mp.Pool(logical_cpus) as pool:
        rv = pool.map(concatenate_partial, cut_alignment_list(alignments))
    pool.join()
    pool.close()

    # TODO: Get path from config file
    conn = sqlite3.connect("job_db.sqlite3")
    for r in rv:
        conn.executemany('''INSERT INTO couplings (pid1, pid2, length, number, finished) VALUES (?, ?, ?, ?, ?)''', r)
    conn.commit()
    conn.close()

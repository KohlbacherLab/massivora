#!/usr/bin/env python3
"""
Export MSA data from zarr format to simple binary format for C++ testing.
Usage: python export_test_data.py <zarr_path> <output_dir>
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from massivora.alignment import BinaryAlignment


def export_msa_binary(zarr_path: str, output_dir: str, reweighting_threshold: float = 0.8):
    """Export MSA and weights to simple binary format."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading alignment from {zarr_path}")
    align = BinaryAlignment(zarr_path)
    
    # Reweight if not already done
    if align.weights is None:
        print(f"Reweighting sequences with threshold {reweighting_threshold}")
        align.Reweight_Sequence(reweighting_threshold)
    
    MSA = np.ascontiguousarray(align.matrix, dtype=np.int32)
    W = np.ascontiguousarray(align.weights, dtype=np.float64)
    Beff = float(align.Beff)
    
    B, N = MSA.shape
    q = int(MSA.max()) + 1
    
    print(f"MSA shape: B={B}, N={N}, q={q}, Beff={Beff:.2f}")
    
    # Write metadata
    meta_path = os.path.join(output_dir, 'meta.bin')
    with open(meta_path, 'wb') as f:
        np.array([B, N, q], dtype=np.int32).tofile(f)
        np.array([Beff], dtype=np.float64).tofile(f)
    print(f"Wrote metadata to {meta_path}")
    
    # Write MSA (row-major, int32)
    msa_path = os.path.join(output_dir, 'msa.bin')
    MSA.tofile(msa_path)
    print(f"Wrote MSA ({MSA.nbytes / 1e6:.2f} MB) to {msa_path}")
    
    # Write weights (float64)
    w_path = os.path.join(output_dir, 'weights.bin')
    W.tofile(w_path)
    print(f"Wrote weights ({W.nbytes / 1e6:.2f} MB) to {w_path}")
    
    print(f"\nExport complete. Use these files for C++ testing.")
    print(f"Expected nParams per site: {q + (N-1) * q * q}")


def main():
    parser = argparse.ArgumentParser(description='Export MSA data for C++ testing')
    parser.add_argument('zarr_path', help='Path to zarr alignment directory')
    parser.add_argument('output_dir', help='Output directory for binary files')
    parser.add_argument('--threshold', type=float, default=0.8, help='Reweighting threshold')
    args = parser.parse_args()
    
    export_msa_binary(args.zarr_path, args.output_dir, args.threshold)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Create shared memory for MSA and W data, then launch C++ optimizer.

Usage:
    python shm_launcher.py <zarr_path_or_bin_dir> <site_id> <optimizer_path>
    
This script:
1. Loads MSA and W data
2. Creates shared memory segments
3. Launches C++ optimizer with shared memory names
4. Reads result from shared memory (optional)
5. Cleans up shared memory
"""

import argparse
import json
import os
import subprocess
import struct
import sys
import time
from multiprocessing import shared_memory
import numpy as np

# Try to import ZarrAlignment for zarr support
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from massivora.alignment import BinaryAlignment
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False


def load_data_from_binary(data_dir: str):
    """Load data from binary files (exported by export_test_data.py)."""
    meta_path = os.path.join(data_dir, 'meta.bin')
    msa_path = os.path.join(data_dir, 'msa.bin')
    w_path = os.path.join(data_dir, 'weights.bin')
    
    with open(meta_path, 'rb') as f:
        B, N, q = struct.unpack('iii', f.read(12))
        Beff = struct.unpack('d', f.read(8))[0]
    
    MSA = np.fromfile(msa_path, dtype=np.int32).reshape(B, N)
    W = np.fromfile(w_path, dtype=np.float64)
    
    return MSA, W, B, N, q, Beff


def load_data_from_zarr(zarr_path: str, reweighting_threshold: float = 0.8):
    """Load data from zarr alignment."""
    if not HAS_ZARR:
        raise ImportError("ZarrAlignment not available")
    
    align = BinaryAlignment(zarr_path)
    if align.weights is None:
        align.Reweight_Sequence(reweighting_threshold)
    
    MSA = np.ascontiguousarray(align.matrix, dtype=np.int32)
    W = np.ascontiguousarray(align.weights, dtype=np.float64)
    B, N = MSA.shape
    q = int(MSA.max()) + 1
    Beff = float(align.Beff)
    
    return MSA, W, B, N, q, Beff


class SharedMSAData:
    """Manages shared memory for MSA data."""
    
    def __init__(self, MSA: np.ndarray, W: np.ndarray, Beff: float, name_prefix: str = "massivora"):
        self.B, self.N = MSA.shape
        self.q = int(MSA.max()) + 1
        self.Beff = Beff
        self.name_prefix = name_prefix
        
        # Shared memory names
        self.meta_name = f"{name_prefix}_meta"
        self.msa_name = f"{name_prefix}_msa"
        self.w_name = f"{name_prefix}_w"
        self.result_name = f"{name_prefix}_result"
        
        # Clean up any existing shared memory with same names
        self._cleanup_existing()
        
        # Create metadata shared memory (B, N, q, Beff)
        # Format: int32 B, int32 N, int32 q, float64 Beff = 20 bytes
        self.meta_shm = shared_memory.SharedMemory(name=self.meta_name, create=True, size=20)
        meta_buf = self.meta_shm.buf
        struct.pack_into('iii', meta_buf, 0, self.B, self.N, self.q)
        struct.pack_into('d', meta_buf, 12, self.Beff)
        
        # Create MSA shared memory
        msa_size = MSA.nbytes
        self.msa_shm = shared_memory.SharedMemory(name=self.msa_name, create=True, size=msa_size)
        msa_arr = np.ndarray(MSA.shape, dtype=np.int32, buffer=self.msa_shm.buf)
        np.copyto(msa_arr, MSA)
        
        # Create W shared memory
        w_size = W.nbytes
        self.w_shm = shared_memory.SharedMemory(name=self.w_name, create=True, size=w_size)
        w_arr = np.ndarray(W.shape, dtype=np.float64, buffer=self.w_shm.buf)
        np.copyto(w_arr, W)
        
        # Create result shared memory (for C++ to write back)
        # Result size: nParams * 8 bytes (float64)
        nParams = self.q + (self.N - 1) * self.q * self.q
        result_size = nParams * 8
        self.result_shm = shared_memory.SharedMemory(name=self.result_name, create=True, size=result_size)
        self.nParams = nParams
        
        print(f"Created shared memory segments:")
        print(f"  Meta:   {self.meta_name} ({20} bytes)")
        print(f"  MSA:    {self.msa_name} ({msa_size / 1e6:.2f} MB)")
        print(f"  W:      {self.w_name} ({w_size / 1e6:.2f} MB)")
        print(f"  Result: {self.result_name} ({result_size / 1e6:.2f} MB)")
    
    def _cleanup_existing(self):
        """Clean up any existing shared memory with same names."""
        for name in [self.meta_name, self.msa_name, self.w_name, self.result_name]:
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
    
    def get_result(self) -> np.ndarray:
        """Read result from shared memory."""
        result = np.ndarray((self.nParams,), dtype=np.float64, buffer=self.result_shm.buf)
        return result.copy()
    
    def cleanup(self):
        """Clean up all shared memory segments."""
        for shm in [self.meta_shm, self.msa_shm, self.w_shm, self.result_shm]:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()


def run_optimizer_with_shm(optimizer_path: str, shm_prefix: str, site_id: int) -> dict:
    """Run C++ optimizer with shared memory."""
    cmd = [optimizer_path, "--shm", shm_prefix, str(site_id)]
    
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall_time = (time.perf_counter() - start) * 1000
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    try:
        timing = json.loads(result.stdout.strip())
        timing['wall_time_ms'] = wall_time
        return timing
    except json.JSONDecodeError:
        print(f"Failed to parse: {result.stdout}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Launch optimizer with shared memory')
    parser.add_argument('data_path', help='Path to zarr alignment or binary data directory')
    parser.add_argument('optimizer_path', help='Path to site_optimizer_shm executable')
    parser.add_argument('--site', type=int, default=0, help='Site ID to compute')
    parser.add_argument('--calls', type=int, default=1, help='Number of calls (reusing shared memory)')
    parser.add_argument('--prefix', default='massivora', help='Shared memory name prefix')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    load_start = time.perf_counter()
    
    if os.path.isfile(os.path.join(args.data_path, 'meta.bin')):
        MSA, W, B, N, q, Beff = load_data_from_binary(args.data_path)
    else:
        MSA, W, B, N, q, Beff = load_data_from_zarr(args.data_path)
    
    load_time = (time.perf_counter() - load_start) * 1000
    print(f"Loaded data in {load_time:.1f}ms: B={B}, N={N}, q={q}, Beff={Beff:.2f}")
    
    # Create shared memory
    print(f"\nCreating shared memory...")
    with SharedMSAData(MSA, W, Beff, args.prefix) as shm_data:
        print(f"\nRunning {args.calls} optimizer calls...")
        
        timings = []
        for i in range(args.calls):
            site = (args.site + i) % N
            timing = run_optimizer_with_shm(args.optimizer_path, args.prefix, site)
            if timing:
                timings.append(timing)
                print(f"  Call {i+1}: site={site}, "
                      f"shm_load={timing.get('shm_load_ms', 0):.1f}ms, "
                      f"compute={timing.get('compute_ms', 0):.1f}ms, "
                      f"total={timing.get('total_ms', 0):.1f}ms, "
                      f"wall={timing.get('wall_time_ms', 0):.1f}ms")
        
        # Summary
        if timings:
            print(f"\n{'='*60}")
            print("SUMMARY (Shared Memory Mode)")
            print(f"{'='*60}")
            
            avg_shm_load = sum(t.get('shm_load_ms', 0) for t in timings) / len(timings)
            avg_compute = sum(t.get('compute_ms', 0) for t in timings) / len(timings)
            avg_total = sum(t.get('total_ms', 0) for t in timings) / len(timings)
            avg_wall = sum(t.get('wall_time_ms', 0) for t in timings) / len(timings)
            
            print(f"Average per call:")
            print(f"  Shared memory mapping:  {avg_shm_load:.2f}ms")
            print(f"  Compute:                {avg_compute:.2f}ms")
            print(f"  Total (C++ side):       {avg_total:.2f}ms")
            print(f"  Wall time:              {avg_wall:.2f}ms")
            print(f"  Process overhead:       {avg_wall - avg_total:.2f}ms")
            
            # Compare with disk I/O version
            print(f"\nComparison with disk I/O (117.5ms load time):")
            disk_total = 117.5 + avg_compute + 9.2  # load + compute + spawn
            shm_total = avg_wall
            speedup = disk_total / shm_total if shm_total > 0 else 0
            print(f"  Disk I/O version:       ~{disk_total:.1f}ms per call")
            print(f"  Shared memory version:  ~{shm_total:.1f}ms per call")
            print(f"  Speedup:                {speedup:.1f}x")
            
            # Projection
            print(f"\nProjection for 1462 sites:")
            disk_time = 1462 * disk_total / 1000 / 60
            shm_time = 1462 * shm_total / 1000 / 60
            print(f"  Disk I/O (sequential):  {disk_time:.1f} min")
            print(f"  Shared memory (seq):    {shm_time:.1f} min")


if __name__ == '__main__':
    main()

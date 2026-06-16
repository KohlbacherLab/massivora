#!/usr/bin/env python3
"""
Test process startup overhead by launching site_optimizer multiple times.

Usage:
    python test_startup_overhead.py <data_dir> <site_optimizer_path> [num_calls]
"""

import argparse
import json
import subprocess
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os


def run_single_site(args):
    """Run site_optimizer for a single site and return timing info."""
    optimizer_path, data_dir, site_id, no_compute = args
    
    cmd = [optimizer_path, data_dir, str(site_id), "--time-only"]
    if no_compute:
        cmd.append("--no-compute")
    
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall_time = (time.perf_counter() - start) * 1000  # ms
    
    if result.returncode != 0:
        print(f"Error for site {site_id}: {result.stderr}")
        return None
    
    try:
        timing = json.loads(result.stdout.strip())
        timing['wall_time_ms'] = wall_time
        timing['overhead_ms'] = wall_time - timing.get('total_ms', 0)
        return timing
    except json.JSONDecodeError:
        print(f"Failed to parse output: {result.stdout}")
        return None


def run_benchmark(optimizer_path: str, data_dir: str, num_calls: int, parallel: int = 1, no_compute: bool = False):
    """Run benchmark and collect statistics."""
    
    print(f"\n{'='*60}")
    print(f"Startup Overhead Benchmark")
    print(f"{'='*60}")
    print(f"Optimizer: {optimizer_path}")
    print(f"Data dir: {data_dir}")
    print(f"Calls: {num_calls}")
    print(f"Parallel workers: {parallel}")
    print(f"No compute: {no_compute}")
    print(f"{'='*60}\n")
    
    # Prepare arguments for all calls
    args_list = [(optimizer_path, data_dir, i % 100, no_compute) for i in range(num_calls)]
    
    results = []
    start_total = time.perf_counter()
    
    if parallel == 1:
        # Sequential execution
        for i, args in enumerate(args_list):
            timing = run_single_site(args)
            if timing:
                results.append(timing)
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{num_calls}")
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            for i, timing in enumerate(executor.map(run_single_site, args_list)):
                if timing:
                    results.append(timing)
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i+1}/{num_calls}")
    
    total_time = time.perf_counter() - start_total
    
    if not results:
        print("No successful results!")
        return
    
    # Calculate statistics
    def stats(values, name):
        if not values:
            return
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        min_v = min(values)
        max_v = max(values)
        print(f"  {name:20s}: mean={mean:8.2f}ms, std={std:6.2f}ms, min={min_v:8.2f}ms, max={max_v:8.2f}ms")
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Successful calls: {len(results)}/{num_calls}")
    print(f"Total wall time: {total_time:.2f}s")
    print(f"Average throughput: {len(results)/total_time:.2f} calls/s")
    print()
    
    # Breakdown
    startup_times = [r['startup_ms'] for r in results]
    load_times = [r['load_ms'] for r in results]
    compute_times = [r['compute_ms'] for r in results if r['compute_ms'] > 0]
    total_times = [r['total_ms'] for r in results]
    wall_times = [r['wall_time_ms'] for r in results]
    overhead_times = [r['overhead_ms'] for r in results]
    
    print("Timing breakdown (from C++ internal measurement):")
    stats(startup_times, "Startup (arg parse)")
    stats(load_times, "Data loading")
    if compute_times:
        stats(compute_times, "Compute")
    stats(total_times, "Total (C++ side)")
    
    print("\nExternal measurement:")
    stats(wall_times, "Wall time (Python)")
    stats(overhead_times, "Process overhead")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    avg_startup = statistics.mean(startup_times)
    avg_load = statistics.mean(load_times)
    avg_compute = statistics.mean(compute_times) if compute_times else 0
    avg_total = statistics.mean(total_times)
    avg_wall = statistics.mean(wall_times)
    avg_overhead = statistics.mean(overhead_times)
    
    print(f"Per-process overhead breakdown:")
    print(f"  C++ startup (before main):  ~{avg_startup:.1f}ms")
    print(f"  Data loading:               ~{avg_load:.1f}ms")
    print(f"  Compute:                    ~{avg_compute:.1f}ms")
    print(f"  Process spawn overhead:     ~{avg_overhead:.1f}ms")
    print(f"  Total per call:             ~{avg_wall:.1f}ms")
    
    if avg_compute > 0:
        efficiency = avg_compute / avg_wall * 100
        print(f"\nCompute efficiency: {efficiency:.1f}% (time spent on actual computation)")
        print(f"Overhead ratio: {(avg_wall - avg_compute) / avg_compute:.2f}x (overhead / compute time)")
    
    # Estimate for full workload
    print(f"\n{'='*60}")
    print("PROJECTION for 1462 sites")
    print(f"{'='*60}")
    
    if avg_compute > 0:
        # Sequential
        seq_time = 1462 * avg_wall / 1000 / 60
        pure_compute = 1462 * avg_compute / 1000 / 60
        print(f"Sequential execution: {seq_time:.1f} min (pure compute: {pure_compute:.1f} min)")
        
        # Parallel with different worker counts
        for workers in [16, 32, 64, 128, 256]:
            # With subprocess, we have process spawn overhead
            parallel_time = 1462 * avg_wall / workers / 1000 / 60
            print(f"Parallel ({workers:3d} workers): {parallel_time:.1f} min")


def main():
    parser = argparse.ArgumentParser(description='Test process startup overhead')
    parser.add_argument('data_dir', help='Directory containing MSA binary files')
    parser.add_argument('optimizer_path', help='Path to site_optimizer executable')
    parser.add_argument('--calls', type=int, default=50, help='Number of calls to make')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--no-compute', action='store_true', help='Skip computation, measure only startup/load')
    args = parser.parse_args()
    
    if not os.path.exists(args.optimizer_path):
        print(f"Error: optimizer not found: {args.optimizer_path}")
        return 1
    
    if not os.path.exists(args.data_dir):
        print(f"Error: data directory not found: {args.data_dir}")
        return 1
    
    run_benchmark(args.optimizer_path, args.data_dir, args.calls, args.parallel, args.no_compute)


if __name__ == '__main__':
    main()

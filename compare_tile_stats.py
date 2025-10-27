#!/usr/bin/env python3
"""
Compare tile distribution statistics between heuristic and uniform modes.

This script analyzes the performance difference between heuristic (workload-balanced)
and uniform (equal distribution) tile distribution modes, focusing on GPU waiting
overhead caused by workload imbalance.

Usage:
    python compare_tile_stats.py <heuristic_log> <uniform_log>

Example:
    python compare_tile_stats.py tile_distribution_stats_heuristic.log tile_distribution_stats_uniform.log
"""

import sys
import re
from pathlib import Path


def parse_log_file(log_path):
    """Parse tile distribution log file and extract statistics."""
    if not Path(log_path).exists():
        print(f"Error: File does not exist: {log_path}")
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Detect mode
    if "Mode: HEURISTIC" in content:
        mode = "heuristic"
    elif "Mode: UNIFORM" in content:
        mode = "uniform"
    else:
        print(f"Error: Could not detect mode in {log_path}")
        return None

    # Extract statistics using regex
    stats = {'mode': mode}

    patterns = {
        'count': r'Call Count\s+(\d+)',
        'mean': r'Mean Time \(ms\)\s+([\d.]+)',
        'median': r'Median Time \(ms\)\s+([\d.]+)',
        'std': r'Std Dev \(ms\)\s+([\d.]+)',
        'min': r'Min Time \(ms\)\s+([\d.]+)',
        'max': r'Max Time \(ms\)\s+([\d.]+)',
        'total': r'Total Time \(ms\)\s+([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            if key == 'count':
                stats[key] = int(match.group(1))
            else:
                stats[key] = float(match.group(1))
        else:
            print(f"Warning: Could not extract {key} from {log_path}")
            stats[key] = 0

    return stats


def compare_tile_stats(heuristic_log, uniform_log):
    """Compare heuristic and uniform tile distribution statistics."""

    print("\n" + "="*80)
    print("TILE DISTRIBUTION MODE COMPARISON ANALYSIS")
    print("="*80)
    print()

    # Parse both log files
    heuristic = parse_log_file(heuristic_log)
    uniform = parse_log_file(uniform_log)

    if not heuristic or not uniform:
        print("Error: Could not parse one or both log files")
        return False

    # Verify modes
    if heuristic['mode'] != 'heuristic':
        print(f"Warning: Expected heuristic mode, got {heuristic['mode']} in {heuristic_log}")
    if uniform['mode'] != 'uniform':
        print(f"Warning: Expected uniform mode, got {uniform['mode']} in {uniform_log}")

    print(f"Heuristic log: {heuristic_log}")
    print(f"Uniform log:   {uniform_log}")
    print()

    # Display comparison table
    print("-"*80)
    print(f"{'Metric':<25} {'Heuristic':>18} {'Uniform':>18} {'Overhead':>18}")
    print("-"*80)

    # Call count
    print(f"{'Call Count':<25} {heuristic['count']:>18,} {uniform['count']:>18,} "
          f"{'-':>18}")

    # Mean time
    h_mean = heuristic['mean']
    u_mean = uniform['mean']
    overhead_mean = u_mean - h_mean
    overhead_pct_mean = (overhead_mean / h_mean * 100) if h_mean > 0 else 0
    print(f"{'Mean Time (ms)':<25} {h_mean:>18.4f} {u_mean:>18.4f} "
          f"{overhead_mean:>+17.4f}")

    # Median time
    h_median = heuristic['median']
    u_median = uniform['median']
    overhead_median = u_median - h_median
    overhead_pct_median = (overhead_median / h_median * 100) if h_median > 0 else 0
    print(f"{'Median Time (ms)':<25} {h_median:>18.4f} {u_median:>18.4f} "
          f"{overhead_median:>+17.4f}")

    # Std dev
    print(f"{'Std Dev (ms)':<25} {heuristic['std']:>18.4f} {uniform['std']:>18.4f} "
          f"{'-':>18}")

    # Min time
    h_min = heuristic['min']
    u_min = uniform['min']
    overhead_min = u_min - h_min
    print(f"{'Min Time (ms)':<25} {h_min:>18.4f} {u_min:>18.4f} "
          f"{overhead_min:>+17.4f}")

    # Max time
    h_max = heuristic['max']
    u_max = uniform['max']
    overhead_max = u_max - h_max
    print(f"{'Max Time (ms)':<25} {h_max:>18.4f} {u_max:>18.4f} "
          f"{overhead_max:>+17.4f}")

    # Total time
    h_total = heuristic['total']
    u_total = uniform['total']
    overhead_total = u_total - h_total
    print(f"{'Total Time (ms)':<25} {h_total:>18.2f} {u_total:>18.2f} "
          f"{overhead_total:>+17.2f}")

    print("-"*80)
    print()

    # Analysis
    print("="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    print()

    print("1. GPU Waiting Overhead (per iteration)")
    print(f"   Average overhead: {overhead_mean:.4f} ms ({overhead_pct_mean:+.2f}%)")
    print(f"   Median overhead:  {overhead_median:.4f} ms ({overhead_pct_median:+.2f}%)")
    print()

    if overhead_mean > 0:
        print(f"   ⚠️  Uniform mode is SLOWER by {overhead_pct_mean:.2f}% on average")
        print(f"   This is due to GPU workload imbalance causing waiting time")
    elif overhead_mean < 0:
        print(f"   ⚡ Uniform mode is FASTER by {-overhead_pct_mean:.2f}% on average")
        print(f"   (Unexpected - check if data is correct)")
    else:
        print(f"   ➡️  Both modes have identical performance")
    print()

    print("2. Total Training Time Impact")
    print(f"   Total overhead: {overhead_total:.2f} ms = {overhead_total/1000:.3f} seconds")

    if heuristic['count'] > 0:
        avg_iterations_per_window = heuristic['count'] / 10  # Assume ~10 windows
        estimated_total_iterations = avg_iterations_per_window * 15  # Assume 15 windows total
        estimated_total_overhead = overhead_mean * estimated_total_iterations
        print(f"   Estimated overhead for full training: {estimated_total_overhead:.2f} ms = {estimated_total_overhead/1000:.3f} seconds")
    print()

    print("3. Workload Balance Analysis")
    h_variance = heuristic['std'] ** 2
    u_variance = uniform['std'] ** 2
    print(f"   Heuristic variance: {h_variance:.4f} ms²")
    print(f"   Uniform variance:   {u_variance:.4f} ms²")

    if u_variance > h_variance * 1.1:
        print(f"   ⚠️  Uniform has {u_variance/h_variance:.2f}x higher variance")
        print(f"   This indicates more unpredictable GPU waiting times")
    elif u_variance < h_variance * 0.9:
        print(f"   ⚡ Uniform has {h_variance/u_variance:.2f}x lower variance")
        print(f"   (Unexpected - check if data is correct)")
    else:
        print(f"   ➡️  Both modes have similar variance")
    print()

    print("4. Summary")
    if overhead_pct_mean > 1.0:
        speedup = 100 / (100 + overhead_pct_mean)
        print(f"   🎯 Heuristic mode achieves {speedup:.2f}x speedup over Uniform mode")
        print(f"   💡 Recommendation: Use HEURISTIC mode for better performance")
    elif overhead_pct_mean < -1.0:
        speedup = (100 - overhead_pct_mean) / 100
        print(f"   🎯 Uniform mode achieves {speedup:.2f}x speedup over Heuristic mode")
        print(f"   💡 Recommendation: Use UNIFORM mode for better performance")
    else:
        print(f"   ➡️  Performance difference is negligible (< 1%)")
        print(f"   💡 Recommendation: Either mode is acceptable")

    print()
    print("="*80)
    print()

    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_tile_stats.py <heuristic_log> <uniform_log>")
        print("Example: python compare_tile_stats.py tile_distribution_stats_heuristic.log tile_distribution_stats_uniform.log")
        sys.exit(1)

    heuristic_log = sys.argv[1]
    uniform_log = sys.argv[2]

    success = compare_tile_stats(heuristic_log, uniform_log)
    sys.exit(0 if success else 1)

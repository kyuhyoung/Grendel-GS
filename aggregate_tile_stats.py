#!/usr/bin/env python3
"""
Aggregate tile distribution statistics from JSON files and generate log file.

This script collects tile distribution statistics from all window directories
and generates a summary log file, even if training was interrupted.

Usage:
    python aggregate_tile_stats.py <output_directory>

Example:
    python aggregate_tile_stats.py ./output/progressive_test
"""

import sys
import json
from pathlib import Path
import numpy as np


def calc_stats(times):
    """Calculate statistics from a list of times (in seconds)."""
    if not times:
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'total': 0.0
        }
    times_ms = [t * 1000 for t in times]  # Convert to milliseconds
    return {
        'count': len(times_ms),
        'mean': np.mean(times_ms),
        'median': np.median(times_ms),
        'std': np.std(times_ms),
        'min': np.min(times_ms),
        'max': np.max(times_ms),
        'total': np.sum(times_ms)
    }


def aggregate_tile_stats(output_dir):
    """Aggregate tile distribution statistics from output directory."""
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"Error: Directory does not exist: {output_path}")
        return False

    # Collect statistics from all window directories
    all_heuristic_times = []
    all_uniform_times = []

    # Find all model directories
    model_dirs = []
    if (output_path / "model_initial").exists():
        model_dirs.append(output_path / "model_initial")

    window_idx = 1
    while (output_path / f"model_window_{window_idx:03d}").exists():
        model_dirs.append(output_path / f"model_window_{window_idx:03d}")
        window_idx += 1

    if not model_dirs:
        print(f"Error: No model directories found in {output_path}")
        return False

    print(f"Found {len(model_dirs)} model directories")

    # Load statistics from each window
    for model_dir in model_dirs:
        stats_file = model_dir / "tile_distribution_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                heuristic_times = stats.get('heuristic_times', [])
                uniform_times = stats.get('uniform_times', [])
                all_heuristic_times.extend(heuristic_times)
                all_uniform_times.extend(uniform_times)
                print(f"  ✓ {model_dir.name}: {len(heuristic_times)} heuristic, {len(uniform_times)} uniform")
            except Exception as e:
                print(f"  ✗ Warning: Could not load {stats_file}: {e}")
        else:
            print(f"  ✗ No stats file in {model_dir.name}")

    # Check if we have any data
    if not all_heuristic_times and not all_uniform_times:
        print("Error: No tile distribution statistics found")
        return False

    # Calculate summary statistics
    heuristic_stats = calc_stats(all_heuristic_times)
    uniform_stats = calc_stats(all_uniform_times)

    # Determine which mode was actually used
    heuristic_used = heuristic_stats['count'] > 0
    uniform_used = uniform_stats['count'] > 0

    # Determine log file name based on mode
    if heuristic_used and not uniform_used:
        mode_name = "heuristic"
    elif uniform_used and not heuristic_used:
        mode_name = "uniform"
    else:
        mode_name = "both"

    log_file_path = output_path / f"tile_distribution_stats_{mode_name}.log"

    # Build output lines
    lines = []
    lines.append("\n" + "="*80)
    lines.append("TILE DISTRIBUTION PERFORMANCE SUMMARY")
    lines.append("="*80)
    lines.append("")

    # Only output statistics for the mode that was actually used
    if heuristic_used and not uniform_used:
        # Only heuristic mode
        lines.append("Mode: HEURISTIC")
        lines.append("-"*80)
        lines.append(f"{'Metric':<30} {'Value':>20}")
        lines.append("-"*80)
        lines.append(f"{'Call Count':<30} {heuristic_stats['count']:>20,}")
        lines.append(f"{'Mean Time (ms)':<30} {heuristic_stats['mean']:>20.4f}")
        lines.append(f"{'Median Time (ms)':<30} {heuristic_stats['median']:>20.4f}")
        lines.append(f"{'Std Dev (ms)':<30} {heuristic_stats['std']:>20.4f}")
        lines.append(f"{'Min Time (ms)':<30} {heuristic_stats['min']:>20.4f}")
        lines.append(f"{'Max Time (ms)':<30} {heuristic_stats['max']:>20.4f}")
        lines.append(f"{'Total Time (ms)':<30} {heuristic_stats['total']:>20.2f}")
    elif uniform_used and not heuristic_used:
        # Only uniform mode
        lines.append("Mode: UNIFORM")
        lines.append("-"*80)
        lines.append(f"{'Metric':<30} {'Value':>20}")
        lines.append("-"*80)
        lines.append(f"{'Call Count':<30} {uniform_stats['count']:>20,}")
        lines.append(f"{'Mean Time (ms)':<30} {uniform_stats['mean']:>20.4f}")
        lines.append(f"{'Median Time (ms)':<30} {uniform_stats['median']:>20.4f}")
        lines.append(f"{'Std Dev (ms)':<30} {uniform_stats['std']:>20.4f}")
        lines.append(f"{'Min Time (ms)':<30} {uniform_stats['min']:>20.4f}")
        lines.append(f"{'Max Time (ms)':<30} {uniform_stats['max']:>20.4f}")
        lines.append(f"{'Total Time (ms)':<30} {uniform_stats['total']:>20.2f}")
    elif heuristic_used and uniform_used:
        # Both modes used - show comparison
        lines.append("Mode: BOTH (Comparison)")
        lines.append("-"*80)
        lines.append(f"{'Metric':<25} {'Heuristic':>18} {'Uniform':>18} {'Difference':>18}")
        lines.append("-"*80)
        lines.append(f"{'Call Count':<25} {heuristic_stats['count']:>18,} {uniform_stats['count']:>18,} "
              f"{abs(heuristic_stats['count'] - uniform_stats['count']):>18,}")

        h_mean = heuristic_stats['mean']
        u_mean = uniform_stats['mean']
        diff_mean = h_mean - u_mean
        lines.append(f"{'Mean Time (ms)':<25} {h_mean:>18.4f} {u_mean:>18.4f} {diff_mean:>+18.4f}")

        h_median = heuristic_stats['median']
        u_median = uniform_stats['median']
        diff_median = h_median - u_median
        lines.append(f"{'Median Time (ms)':<25} {h_median:>18.4f} {u_median:>18.4f} {diff_median:>+18.4f}")

        lines.append(f"{'Std Dev (ms)':<25} {heuristic_stats['std']:>18.4f} {uniform_stats['std']:>18.4f} {'─':>18}")

        h_min = heuristic_stats['min']
        u_min = uniform_stats['min']
        diff_min = h_min - u_min
        lines.append(f"{'Min Time (ms)':<25} {h_min:>18.4f} {u_min:>18.4f} {diff_min:>+18.4f}")

        h_max = heuristic_stats['max']
        u_max = uniform_stats['max']
        diff_max = h_max - u_max
        lines.append(f"{'Max Time (ms)':<25} {h_max:>18.4f} {u_max:>18.4f} {diff_max:>+18.4f}")

        h_total = heuristic_stats['total']
        u_total = uniform_stats['total']
        diff_total = h_total - u_total
        lines.append(f"{'Total Time (ms)':<25} {h_total:>18.2f} {u_total:>18.2f} {diff_total:>+18.2f}")
        lines.append("-"*80)

        # Performance comparison
        if h_mean > 0:
            speedup = u_mean / h_mean
            if speedup > 1.01:
                lines.append(f"⚡ Heuristic is {speedup:.2f}x FASTER than Uniform (on average)")
            elif speedup < 0.99:
                lines.append(f"⚠️  Uniform is {1/speedup:.2f}x FASTER than Heuristic (on average)")
            else:
                lines.append(f"➡️  Both modes have similar performance (difference < 1%)")

    lines.append("="*80 + "\n")

    # Print to console
    for line in lines:
        print(line)

    # Save to log file
    with open(log_file_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"💾 Saved tile distribution summary to: {log_file_path}\n")

    # Save aggregated summary JSON
    summary_file = output_path / f"tile_distribution_stats_summary_{mode_name}.json"
    summary_data = {
        'mode': mode_name,
        'heuristic': heuristic_stats,
        'uniform': uniform_stats,
        'all_heuristic_times': all_heuristic_times,
        'all_uniform_times': all_uniform_times
    }
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"💾 Saved aggregated statistics to: {summary_file}\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not save summary statistics: {e}\n")

    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python aggregate_tile_stats.py <output_directory>")
        print("Example: python aggregate_tile_stats.py ./output/progressive_test")
        sys.exit(1)

    output_dir = sys.argv[1]
    success = aggregate_tile_stats(output_dir)
    sys.exit(0 if success else 1)

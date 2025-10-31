#!/usr/bin/env python3
"""
Merge final gaussians and removed gaussians PLY files from progressive training
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

def load_ply_gaussians(ply_path):
    """Load gaussians from PLY file"""
    print(f"📄 Loading PLY: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # Convert to numpy array
    vertex_array = np.array(vertex.data)

    # Extract all properties
    gaussians = {}
    for prop in vertex_array.dtype.names:
        gaussians[prop] = vertex_array[prop]

    print(f"   ✅ Loaded {len(vertex_array)} gaussians")
    return gaussians

def merge_gaussians(gaussian_list):
    """Merge multiple gaussian dictionaries"""
    if not gaussian_list:
        raise ValueError("No gaussians to merge")

    if len(gaussian_list) == 1:
        return gaussian_list[0]

    print(f"🔗 Merging {len(gaussian_list)} PLY files...")

    # Get all property names from first gaussian set
    all_props = set(gaussian_list[0].keys())

    # Verify all gaussian sets have same properties
    for i, gaussians in enumerate(gaussian_list[1:], 1):
        if set(gaussians.keys()) != all_props:
            print(f"⚠️  Warning: PLY file {i+1} has different properties")
            all_props = all_props.intersection(set(gaussians.keys()))

    print(f"📋 Common properties: {sorted(all_props)}")

    # Merge each property
    merged = {}
    total_gaussians = sum(len(gaussians[list(all_props)[0]]) for gaussians in gaussian_list)
    print(f"📊 Total gaussians after merge: {total_gaussians}")

    for prop in all_props:
        prop_arrays = [gaussians[prop] for gaussians in gaussian_list]
        merged[prop] = np.concatenate(prop_arrays)
        print(f"   {prop}: {merged[prop].shape}")

    return merged

def save_merged_ply(gaussians, output_path):
    """Save merged gaussians to PLY file"""
    print(f"💾 Saving merged PLY: {output_path}")

    # Prepare vertex data
    vertex_data = []
    dtypes = []

    for prop, values in gaussians.items():
        if values.ndim == 1:
            vertex_data.append(values)
            dtypes.append((prop, values.dtype))
        else:
            # Multi-dimensional properties (like SH coefficients)
            for i in range(values.shape[1]):
                if prop.startswith('f_dc_') or prop.startswith('f_rest_'):
                    # SH coefficients already have correct naming
                    vertex_data.append(values[:, i])
                    dtypes.append((f"{prop}_{i}" if not prop.endswith(f"_{i}") else prop, values.dtype))
                else:
                    vertex_data.append(values[:, i])
                    dtypes.append((f"{prop}_{i}", values.dtype))

    # Create structured array
    vertex_array = np.empty(len(gaussians[list(gaussians.keys())[0]]), dtype=dtypes)

    prop_idx = 0
    for prop, values in gaussians.items():
        if values.ndim == 1:
            vertex_array[dtypes[prop_idx][0]] = values
            prop_idx += 1
        else:
            for i in range(values.shape[1]):
                dtype_name = dtypes[prop_idx][0]
                vertex_array[dtype_name] = values[:, i]
                prop_idx += 1

    # Create PLY element
    vertex_element = PlyElement.describe(vertex_array, 'vertex')

    # Write PLY file
    PlyData([vertex_element], text=False).write(output_path)

    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"✅ Merged PLY saved!")
    print(f"   📄 File: {output_path}")
    print(f"   📏 Size: {file_size:.2f} MB")
    print(f"   🔢 Gaussians: {len(vertex_array):,}")

def main():
    parser = argparse.ArgumentParser(description='Merge progressive training PLY files')
    parser.add_argument('--output_dir', required=True, help='Progressive training output directory')
    parser.add_argument('--output_ply', required=True, help='Output merged PLY file path')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("🔍 Searching for PLY files...")

    # Find final gaussians PLY
    final_ply = None
    if (output_dir / "final_gaussians_new.ply").exists():
        final_ply = output_dir / "final_gaussians_new.ply"
    elif (output_dir / "final_gaussians.ply").exists():
        final_ply = output_dir / "final_gaussians.ply"
    else:
        # Look for latest window checkpoint and export it
        window_dirs = sorted([d for d in output_dir.iterdir() if d.name.startswith("model_window_")])
        if window_dirs:
            latest_window = window_dirs[-1]
            checkpoint_dir = latest_window / "checkpoints"
            if checkpoint_dir.exists():
                print(f"🔄 Exporting final gaussians from {latest_window.name}...")
                final_ply = output_dir / "final_gaussians.ply"

                # Use simple_export_ply.py to extract
                import subprocess
                result = subprocess.run([
                    sys.executable, "simple_export_ply.py",
                    "--checkpoint_dir", str(checkpoint_dir),
                    "--output_ply", str(final_ply)
                ], capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"❌ Failed to export final gaussians: {result.stderr}")
                    return

    if not final_ply or not final_ply.exists():
        print("❌ No final gaussians PLY found")
        return

    # Find removed gaussians PLY files
    removed_plys = list(output_dir.glob("**/removed_gaussians_*.ply"))

    print(f"📋 Found PLY files:")
    print(f"   🎯 Final gaussians: {final_ply}")
    print(f"   🗑️  Removed gaussians: {len(removed_plys)} files")
    for removed_ply in removed_plys:
        print(f"      - {removed_ply}")

    # Load all PLY files
    all_gaussians = []

    # Load final gaussians
    final_gaussians = load_ply_gaussians(final_ply)
    all_gaussians.append(final_gaussians)

    # Load removed gaussians
    for removed_ply in removed_plys:
        removed_gaussians = load_ply_gaussians(removed_ply)
        all_gaussians.append(removed_gaussians)

    if len(all_gaussians) == 1:
        print("ℹ️  Only final gaussians found, no removed gaussians to merge")
        # Just copy the final PLY
        import shutil
        shutil.copy2(final_ply, args.output_ply)
        print(f"📄 Copied final gaussians to: {args.output_ply}")
        return

    # Merge all gaussians
    merged_gaussians = merge_gaussians(all_gaussians)

    # Save merged PLY
    save_merged_ply(merged_gaussians, args.output_ply)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

"""
Usage example for the new --sparse_path option that allows images and sparse folders
to be in separate locations.

Examples:

1. Standard usage (images and sparse in same parent directory):
   python train.py --source_path /data/scene --images images

2. Separate paths usage:
   python train.py --source_path /data/scene --images /data/scene/images --sparse_path /data/scene/temp_initial/sparse/0

This fixes the FileNotFoundError when sparse folder is in a different location than images folder.
"""

print("New --sparse_path option added!")
print("\nUsage examples:")
print("1. Standard (backward compatible):")
print("   python train.py --source_path /data/scene --images images")
print("\n2. Separate paths (new functionality):")
print("   python train.py --source_path /data/scene --images /path/to/images --sparse_path /path/to/sparse/0")
print("\nThis solves the issue where images and sparse folders are not at the same level.")
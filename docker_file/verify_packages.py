#!/usr/bin/env python
"""
Verify that all required packages are installed and importable
"""

import sys
import importlib

packages = [
    'torch',
    'gsplat', 
    'simple_knn',
    'diff_gaussian_rasterization'
]

print("=" * 60)
print("PACKAGE VERIFICATION")
print("=" * 60)

all_success = True

for package in packages:
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {package:30} imported successfully (version: {version})")
    except ImportError as e:
        print(f"✗ {package:30} FAILED: {e}")
        all_success = False

print("=" * 60)

if all_success:
    print("SUCCESS: All packages imported successfully")
    sys.exit(0)
else:
    print("FAILURE: Some packages failed to import")
    sys.exit(1)
#!/usr/bin/env python3
"""
Adaptive Tile Training Script

This script runs the adaptive tile-based 3DGS training.
It automatically splits tiles when OOM occurs and ensures
fair training order across all tiles.

Usage:
    # Single GPU
    python scripts/train_adaptive.py \
        --colmap_path data/scene/sparse/0 \
        --images_path data/scene/images \
        --output_path output/scene_adaptive

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 scripts/train_adaptive.py \
        --colmap_path data/scene/sparse/0 \
        --images_path data/scene/images \
        --output_path output/scene_adaptive \
        --distributed
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Grendel-GS"))

from src.adaptive_trainer import main

if __name__ == "__main__":
    main()

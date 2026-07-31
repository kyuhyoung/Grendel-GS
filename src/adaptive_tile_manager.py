"""
Adaptive Tile Manager for OOM-aware 3DGS Training

This module implements an adaptive tile splitting strategy that:
1. Starts with the entire scene as a single tile
2. Splits tiles when OOM occurs
3. Prioritizes tiles based on split count (lower = higher priority)
4. Ensures fair training order across all tiles
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum
from pathlib import Path
import json
import numpy as np

from .tile_storage import BBox


class TileStatus(Enum):
    UNDONE = "UNDONE"
    DONE = "DONE"


@dataclass
class AdaptiveTile:
    """Represents a single adaptive tile with metadata."""
    tile_id: str
    bbox: BBox
    status: TileStatus
    priority: int  # Lower number = higher priority (split count)
    area: float
    gaussians_path: Optional[str] = None  # Path to saved gaussians PLY

    def to_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "bbox": self.bbox.to_dict(),
            "status": self.status.value,
            "priority": self.priority,
            "area": self.area,
            "gaussians_path": self.gaussians_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveTile":
        return cls(
            tile_id=d["tile_id"],
            bbox=BBox.from_dict(d["bbox"]),
            status=TileStatus(d["status"]),
            priority=d["priority"],
            area=d["area"],
            gaussians_path=d.get("gaussians_path"),
        )


class AdaptiveTileManager:
    """
    Manages adaptive tile splitting and training order.

    The manager starts with the entire scene as a single tile and
    dynamically splits tiles when OOM occurs during training.
    """

    def __init__(self, initial_bbox: Optional[BBox] = None, save_dir: Optional[Path] = None):
        """
        Initialize the adaptive tile manager.

        Args:
            initial_bbox: Initial bounding box covering the entire scene.
                         If None, must call load_state() or set_initial_bbox() later.
            save_dir: Directory to save/load manager state and tile data.
        """
        self.tiles: Dict[str, AdaptiveTile] = {}
        self.tile_counter = 0
        self.save_dir = Path(save_dir) if save_dir else None
        self.current_tile_id: Optional[str] = None
        self.current_area: float = 0.0

        if initial_bbox is not None:
            self._create_initial_tile(initial_bbox)

    def _create_initial_tile(self, bbox: BBox):
        """Create the initial tile covering the entire scene."""
        initial_tile = AdaptiveTile(
            tile_id=self._new_id(),
            bbox=bbox,
            status=TileStatus.UNDONE,
            priority=0,
            area=self._calc_area(bbox)
        )
        self.tiles[initial_tile.tile_id] = initial_tile
        self.current_tile_id = initial_tile.tile_id
        self.current_area = initial_tile.area

    def set_initial_bbox(self, bbox: BBox):
        """Set the initial bounding box if not provided in constructor."""
        if self.tiles:
            raise RuntimeError("Cannot set initial bbox: tiles already exist")
        self._create_initial_tile(bbox)

    def _new_id(self) -> str:
        """Generate a new unique tile ID."""
        self.tile_counter += 1
        return f"tile_{self.tile_counter:08d}"

    def _calc_area(self, bbox: BBox) -> float:
        """Calculate the XY area of a bounding box (ignoring Z)."""
        size = bbox.max - bbox.min
        return float(size[0] * size[1])

    def get_current_tile(self) -> Optional[AdaptiveTile]:
        """Get the current tile being processed."""
        if self.current_tile_id is None:
            return None
        return self.tiles.get(self.current_tile_id)

    def split_tile(self, tile_id: str) -> Tuple[str, str]:
        """
        Split a tile into two along its longer axis.

        The first tile (B) keeps the same priority.
        The second tile (C) gets priority + 1.

        Args:
            tile_id: ID of the tile to split.

        Returns:
            Tuple of (B_tile_id, C_tile_id)
        """
        if tile_id not in self.tiles:
            raise ValueError(f"Tile {tile_id} not found")

        tile = self.tiles[tile_id]
        bbox = tile.bbox
        size = bbox.max - bbox.min

        # Split along the longer axis (X or Y)
        if size[0] >= size[1]:
            # X is longer, split along X
            mid_x = (bbox.min[0] + bbox.max[0]) / 2
            bbox_b = BBox(
                min_xyz=bbox.min.copy(),
                max_xyz=np.array([mid_x, bbox.max[1], bbox.max[2]])
            )
            bbox_c = BBox(
                min_xyz=np.array([mid_x, bbox.min[1], bbox.min[2]]),
                max_xyz=bbox.max.copy()
            )
        else:
            # Y is longer, split along Y
            mid_y = (bbox.min[1] + bbox.max[1]) / 2
            bbox_b = BBox(
                min_xyz=bbox.min.copy(),
                max_xyz=np.array([bbox.max[0], mid_y, bbox.max[2]])
            )
            bbox_c = BBox(
                min_xyz=np.array([bbox.min[0], mid_y, bbox.min[2]]),
                max_xyz=bbox.max.copy()
            )

        # Create new tiles
        # B: keeps current priority
        tile_b = AdaptiveTile(
            tile_id=self._new_id(),
            bbox=bbox_b,
            status=TileStatus.UNDONE,
            priority=tile.priority,
            area=self._calc_area(bbox_b)
        )

        # C: priority + 1 (will be processed later)
        tile_c = AdaptiveTile(
            tile_id=self._new_id(),
            bbox=bbox_c,
            status=TileStatus.UNDONE,
            priority=tile.priority + 1,
            area=self._calc_area(bbox_c)
        )

        # Remove original tile, add new tiles
        del self.tiles[tile_id]
        self.tiles[tile_b.tile_id] = tile_b
        self.tiles[tile_c.tile_id] = tile_c

        print(f"[AdaptiveTileManager] Split {tile_id} into {tile_b.tile_id} (priority={tile_b.priority}) "
              f"and {tile_c.tile_id} (priority={tile_c.priority})")

        return tile_b.tile_id, tile_c.tile_id

    def mark_done(self, tile_id: str):
        """Mark a tile as done (training completed)."""
        if tile_id not in self.tiles:
            raise ValueError(f"Tile {tile_id} not found")
        self.tiles[tile_id].status = TileStatus.DONE
        print(f"[AdaptiveTileManager] Marked {tile_id} as DONE")

    def set_gaussians_path(self, tile_id: str, path: str):
        """Set the path where tile's gaussians are saved."""
        if tile_id not in self.tiles:
            raise ValueError(f"Tile {tile_id} not found")
        self.tiles[tile_id].gaussians_path = path

    def find_undone_tile_by_area(self, target_area: float, tolerance: float = 0.05) -> Optional[str]:
        """
        Find an UNDONE tile with the specified area.

        If multiple tiles match, returns the one with lowest priority (longest waiting).

        Args:
            target_area: Target area to match.
            tolerance: Relative tolerance for area matching (default 5%).

        Returns:
            Tile ID if found, None otherwise.
        """
        candidates = [
            t for t in self.tiles.values()
            if t.status == TileStatus.UNDONE
            and abs(t.area - target_area) / max(target_area, 1e-6) < tolerance
        ]

        if not candidates:
            return None

        # Sort by priority (lowest first), then by tile_id for determinism
        candidates.sort(key=lambda t: (t.priority, t.tile_id))
        return candidates[0].tile_id

    def find_any_undone_tile(self) -> Optional[str]:
        """
        Find any UNDONE tile, prioritizing by lowest priority number.

        Returns:
            Tile ID if found, None otherwise.
        """
        candidates = [
            t for t in self.tiles.values()
            if t.status == TileStatus.UNDONE
        ]

        if not candidates:
            return None

        candidates.sort(key=lambda t: (t.priority, t.tile_id))
        return candidates[0].tile_id

    def get_next_tile(self) -> Optional[str]:
        """
        Get the next tile to process based on the adaptive algorithm.

        Priority order:
        1. Same area as current, lowest priority number
        2. Half area of current, lowest priority number
        3. Any UNDONE tile with lowest priority number

        Returns:
            Tile ID if found, None if all tiles are done.
        """
        # Try to find tile with same area
        D = self.find_undone_tile_by_area(self.current_area)
        if D:
            self.current_tile_id = D
            return D

        # Try to find tile with half area
        E = self.find_undone_tile_by_area(self.current_area / 2)
        if E:
            self.current_tile_id = E
            self.current_area = self.current_area / 2
            return E

        # Try any remaining UNDONE tile
        remaining = self.find_any_undone_tile()
        if remaining:
            self.current_tile_id = remaining
            self.current_area = self.tiles[remaining].area
            return remaining

        return None

    def handle_oom(self, tile_id: str) -> str:
        """
        Handle OOM error by splitting the tile.

        Args:
            tile_id: ID of the tile that caused OOM.

        Returns:
            ID of the first split tile (B) to continue training.
        """
        B, C = self.split_tile(tile_id)
        self.current_tile_id = B
        self.current_area = self.tiles[B].area
        return B

    def handle_success(self, tile_id: str) -> Optional[str]:
        """
        Handle successful training of a tile.

        Args:
            tile_id: ID of the successfully trained tile.

        Returns:
            ID of the next tile to process, or None if all done.
        """
        self.mark_done(tile_id)
        self.current_area = self.tiles[tile_id].area
        return self.get_next_tile()

    def is_all_done(self) -> bool:
        """Check if all tiles have been trained."""
        return all(t.status == TileStatus.DONE for t in self.tiles.values())

    def get_statistics(self) -> Dict:
        """Get statistics about the current state."""
        done_tiles = [t for t in self.tiles.values() if t.status == TileStatus.DONE]
        undone_tiles = [t for t in self.tiles.values() if t.status == TileStatus.UNDONE]

        return {
            "total_tiles": len(self.tiles),
            "done_tiles": len(done_tiles),
            "undone_tiles": len(undone_tiles),
            "done_area": sum(t.area for t in done_tiles),
            "undone_area": sum(t.area for t in undone_tiles),
            "max_priority": max((t.priority for t in self.tiles.values()), default=0),
            "current_tile": self.current_tile_id,
            "current_area": self.current_area,
        }

    def save_state(self, path: Optional[Path] = None):
        """Save manager state to JSON file."""
        if path is None:
            if self.save_dir is None:
                raise ValueError("No save path specified")
            path = self.save_dir / "adaptive_tile_state.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "tile_counter": self.tile_counter,
            "current_tile_id": self.current_tile_id,
            "current_area": self.current_area,
            "tiles": {tid: t.to_dict() for tid, t in self.tiles.items()},
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        print(f"[AdaptiveTileManager] State saved to {path}")

    def load_state(self, path: Optional[Path] = None):
        """Load manager state from JSON file."""
        if path is None:
            if self.save_dir is None:
                raise ValueError("No load path specified")
            path = self.save_dir / "adaptive_tile_state.json"

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        with open(path, "r") as f:
            state = json.load(f)

        self.tile_counter = state["tile_counter"]
        self.current_tile_id = state["current_tile_id"]
        self.current_area = state["current_area"]
        self.tiles = {
            tid: AdaptiveTile.from_dict(tdata)
            for tid, tdata in state["tiles"].items()
        }

        print(f"[AdaptiveTileManager] State loaded from {path}")
        stats = self.get_statistics()
        print(f"  - Total tiles: {stats['total_tiles']}, Done: {stats['done_tiles']}, Undone: {stats['undone_tiles']}")

    def print_status(self):
        """Print current status of all tiles."""
        print("\n" + "=" * 60)
        print("Adaptive Tile Manager Status")
        print("=" * 60)

        stats = self.get_statistics()
        print(f"Total: {stats['total_tiles']} tiles, Done: {stats['done_tiles']}, Undone: {stats['undone_tiles']}")
        print(f"Current tile: {stats['current_tile']}, Current area: {stats['current_area']:.2f}")
        print(f"Max split depth: {stats['max_priority']}")
        print("-" * 60)

        # Sort by priority then tile_id
        sorted_tiles = sorted(self.tiles.values(), key=lambda t: (t.priority, t.tile_id))

        for tile in sorted_tiles:
            status_marker = "✓" if tile.status == TileStatus.DONE else "○"
            current_marker = "→" if tile.tile_id == self.current_tile_id else " "
            print(f"{current_marker} {status_marker} {tile.tile_id}: priority={tile.priority}, "
                  f"area={tile.area:.2f}, status={tile.status.value}")

        print("=" * 60 + "\n")

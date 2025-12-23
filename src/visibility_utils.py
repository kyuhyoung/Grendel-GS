"""Utilities for working with precomputed tile visibility metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TileObservation:
    tile_id: str
    coord: Tuple[int, int, int]
    num_gaussians: int
    camera_index: int
    image_name: str
    bbox: Tuple[int, int, int, int]
    pixel_area: float
    image_width: int
    image_height: int


@dataclass
class CameraVisibility:
    index: int
    image_name: str
    cam_id: int
    tiles: List[TileObservation]


@dataclass
class TileVisibility:
    tile_id: str
    coord: Tuple[int, int, int]
    num_gaussians: int
    cameras: List[TileObservation]


@dataclass
class VisibilityDataset:
    summary: Dict
    tiles: Dict[str, TileVisibility]
    cameras: Dict[int, CameraVisibility]


def load_visibility_metadata(path: Path) -> VisibilityDataset:
    """Load tile visibility metadata written by precompute_tile_visibility."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})

    tiles: Dict[str, TileVisibility] = {}
    raw_tiles = data.get("tiles", {})
    for tile_id, payload in raw_tiles.items():
        coord = tuple(payload.get("coord", [0, 0, 0]))
        num_gaussians = int(payload.get("num_gaussians", 0))
        cameras_payload = payload.get("cameras", [])
        cameras: List[TileObservation] = []
        for cam in cameras_payload:
            obs = TileObservation(
                tile_id=tile_id,
                coord=coord,
                num_gaussians=num_gaussians,
                camera_index=int(cam.get("camera_index", -1)),
                image_name=cam.get("image_name", ""),
                bbox=tuple(cam.get("bbox", [0, 0, 0, 0])),
                pixel_area=float(cam.get("pixel_area", 0.0)),
                image_width=int(cam.get("width", 0)),
                image_height=int(cam.get("height", 0)),
            )
            cameras.append(obs)
        tiles[tile_id] = TileVisibility(
            tile_id=tile_id,
            coord=coord,
            num_gaussians=num_gaussians,
            cameras=cameras,
        )

    cameras: Dict[int, CameraVisibility] = {}
    raw_cameras = data.get("cameras", {})
    for key, payload in raw_cameras.items():
        try:
            cam_index = int(key)
        except ValueError:
            cam_index = int(payload.get("index", -1))
        tiles_list = []
        for tile_payload in payload.get("tiles", []):
            tile_id = tile_payload.get("tile_id")
            coord = tuple(tile_payload.get("coord", [0, 0, 0]))
            num_gaussians = int(tile_payload.get("num_gaussians", 0))
            obs = TileObservation(
                tile_id=tile_id,
                coord=coord,
                num_gaussians=num_gaussians,
                camera_index=cam_index,
                image_name=payload.get("image_name", ""),
                bbox=tuple(tile_payload.get("bbox", [0, 0, 0, 0])),
                pixel_area=float(tile_payload.get("pixel_area", 0.0)),
                image_width=int(tile_payload.get("width", 0)),
                image_height=int(tile_payload.get("height", 0)),
            )
            tiles_list.append(obs)
        cameras[cam_index] = CameraVisibility(
            index=cam_index,
            image_name=payload.get("image_name", ""),
            cam_id=int(payload.get("cam_id", -1)),
            tiles=tiles_list,
        )

    return VisibilityDataset(summary=summary, tiles=tiles, cameras=cameras)


def compute_tile_statistics(dataset: VisibilityDataset) -> List[Dict[str, float]]:
    """Return per-tile statistics for prioritization."""

    stats = []
    for tile in dataset.tiles.values():
        num_views = len(tile.cameras)
        total_pixel_area = sum(obs.pixel_area for obs in tile.cameras)
        avg_pixel_area = total_pixel_area / num_views if num_views > 0 else 0.0
        stats.append(
            {
                "tile_id": tile.tile_id,
                "coord": tile.coord,
                "num_gaussians": tile.num_gaussians,
                "num_views": num_views,
                "total_pixel_area": total_pixel_area,
                "avg_pixel_area": avg_pixel_area,
            }
        )
    return stats


def compute_camera_statistics(dataset: VisibilityDataset) -> List[Dict[str, float]]:
    """Return per-camera statistics, useful for scheduling."""

    stats = []
    for cam in dataset.cameras.values():
        num_tiles = len(cam.tiles)
        total_pixel_area = sum(obs.pixel_area for obs in cam.tiles)
        avg_pixel_area = total_pixel_area / num_tiles if num_tiles > 0 else 0.0
        stats.append(
            {
                "camera_index": cam.index,
                "image_name": cam.image_name,
                "cam_id": cam.cam_id,
                "num_tiles": num_tiles,
                "total_pixel_area": total_pixel_area,
                "avg_pixel_area": avg_pixel_area,
            }
        )
    stats.sort(key=lambda x: x["camera_index"])
    return stats


def _spread_bits(n: int) -> int:
    result = 0
    bit_index = 0
    while n:
        if n & 1:
            result |= 1 << bit_index
        n >>= 1
        bit_index += 3
    return result


def _morton_code(coord: Tuple[int, int, int]) -> int:
    x, y, z = coord
    return _spread_bits(x) | (_spread_bits(y) << 1) | (_spread_bits(z) << 2)


def _row_major_code(coord: Tuple[int, int, int]) -> int:
    x, y, z = coord
    return (z << 40) + (y << 20) + x


def _spatial_rank(coord: Tuple[int, int, int], method: str) -> int:
    if method == "morton":
        return _morton_code(coord)
    if method == "row-major":
        return _row_major_code(coord)
    raise ValueError(f"Unknown spatial ordering method: {method}")


@dataclass
class TileBatchPlan:
    batch_id: int
    tiles: List[str]
    cameras: List[int]
    total_pixel_area: float
    num_gaussians: int


@dataclass
class PlannedBatch:
    meta: TileBatchPlan
    tile_details: List[TileObservation]
    camera_details: List[TileObservation]


def plan_batches(
    dataset: VisibilityDataset,
    max_tiles: int,
    max_total_pixel_area: Optional[float] = None,
    order_strategy: str = "camera",
    spatial_order: str = "morton",
) -> List[PlannedBatch]:
    """Plan tile batches using either camera-first or tile-first ordering."""

    batches: List[PlannedBatch] = []
    current_tiles: Dict[str, TileObservation] = {}
    current_cameras: Dict[int, List[TileObservation]] = {}
    current_pixel_area = 0.0
    current_gaussians = 0
    batch_id = 0

    def flush() -> None:
        nonlocal current_tiles, current_cameras, current_pixel_area, current_gaussians, batch_id
        if not current_tiles and not current_cameras:
            return
        meta = TileBatchPlan(
            batch_id=batch_id,
            tiles=list(current_tiles.keys()),
            cameras=sorted(current_cameras.keys()),
            total_pixel_area=current_pixel_area,
            num_gaussians=current_gaussians,
        )
        tile_details = list(current_tiles.values())
        camera_details = [obs for patches in current_cameras.values() for obs in patches]
        batches.append(PlannedBatch(meta=meta, tile_details=tile_details, camera_details=camera_details))
        batch_id += 1
        current_tiles = {}
        current_cameras = {}
        current_pixel_area = 0.0
        current_gaussians = 0

    if order_strategy == "camera":
        for cam_index in sorted(dataset.cameras.keys()):
            camera = dataset.cameras[cam_index]
            if not camera.tiles:
                continue
            for obs in sorted(camera.tiles, key=lambda o: o.pixel_area, reverse=True):
                new_tiles_set = set(current_tiles.keys())
                new_tiles_set.add(obs.tile_id)
                projected_tile_count = len(new_tiles_set)
                projected_pixel_area = current_pixel_area + obs.pixel_area
                if max_tiles > 0 and projected_tile_count > max_tiles:
                    flush()
                elif max_total_pixel_area is not None and projected_pixel_area > max_total_pixel_area:
                    flush()
                current_tiles.setdefault(obs.tile_id, obs)
                current_pixel_area += obs.pixel_area
                current_gaussians += obs.num_gaussians
                patches = current_cameras.setdefault(cam_index, [])
                patches.append(obs)
        flush()
        return batches

    if order_strategy == "tile":
        tile_priority: List[Tuple[float, float, int, str, TileVisibility]] = []
        for tile in dataset.tiles.values():
            if not tile.cameras:
                continue
            total_pixel_area = sum(obs.pixel_area for obs in tile.cameras)
            num_views = len(tile.cameras)
            spatial_key = _spatial_rank(tile.coord, spatial_order)
            tile_priority.append((-total_pixel_area, -num_views, spatial_key, tile.tile_id, tile))

        tile_priority.sort()
        scheduled: set[str] = set()

        for _, _, _, _, tile in tile_priority:
            if tile.tile_id in scheduled:
                continue
            observations = tile.cameras
            tile_pixel_area = sum(obs.pixel_area for obs in observations)

            if current_tiles:
                projected_tiles = len(current_tiles) + 1
                projected_pixel_area = current_pixel_area + tile_pixel_area
                need_flush = False
                if max_tiles > 0 and projected_tiles > max_tiles:
                    need_flush = True
                elif max_total_pixel_area is not None and projected_pixel_area > max_total_pixel_area:
                    need_flush = True
                if need_flush:
                    flush()

            current_tiles[tile.tile_id] = observations[0]
            scheduled.add(tile.tile_id)
            current_pixel_area += tile_pixel_area
            current_gaussians += tile.num_gaussians
            for obs in observations:
                patches = current_cameras.setdefault(obs.camera_index, [])
                patches.append(obs)

        flush()
        return batches

    raise ValueError(f"Unsupported order_strategy: {order_strategy}")

#!/usr/bin/env python3
"""Precompute image↔tile visibility metadata using COLMAP poses."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Grendel-GS"))

from scene.colmap_loader import qvec2rotmat
from src.tile_storage import TileStorage
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


def _load_colmap_cameras(colmap_sparse_dir: Path) -> Dict[int, Dict]:
    cameras_txt = colmap_sparse_dir / "cameras.txt"
    cameras: Dict[int, Dict] = {}
    with open(cameras_txt, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            if model == "PINHOLE":
                fx, fy, cx, cy = params
            elif model == "SIMPLE_PINHOLE":
                f, cx, cy = params
                fx = fy = f
            else:
                raise ValueError(f"Unsupported COLMAP camera model: {model}")

            fovx = focal2fov(fx, width)
            fovy = focal2fov(fy, height)

            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "fovx": fovx,
                "fovy": fovy,
            }
    if not cameras:
        raise RuntimeError(f"No cameras parsed from {cameras_txt}")
    return cameras


def _load_colmap_images(colmap_sparse_dir: Path) -> List[Dict]:
    images_txt = colmap_sparse_dir / "images.txt"
    views: List[Dict] = []
    with open(images_txt, "r") as f:
        idx = 0
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            image_name = parts[9]
            views.append(
                {
                    "index": idx,
                    "cam_id": cam_id,
                    "qvec": np.array([qw, qx, qy, qz], dtype=np.float64),
                    "tvec": np.array([tx, ty, tz], dtype=np.float64),
                    "image_name": image_name,
                }
            )
            idx += 1
    if not views:
        raise RuntimeError(f"No images parsed from {images_txt}")
    return views


def _project_points(
    full_proj: np.ndarray,
    width: int,
    height: int,
    points: np.ndarray,
    padding_ratio: float,
) -> Optional[Tuple[List[int], float]]:
    if points.size == 0:
        return None

    proj = full_proj.astype(np.float64, copy=False)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([points.astype(np.float64, copy=False), ones], axis=1)
    clip = pts_h @ proj
    w = clip[:, 3]
    valid = w > 0
    if not np.any(valid):
        return None
    clip = clip[valid]
    w = w[valid]
    ndc = clip[:, :3] / w[:, None]
    inside = (
        (ndc[:, 0] >= -1.0)
        & (ndc[:, 0] <= 1.0)
        & (ndc[:, 1] >= -1.0)
        & (ndc[:, 1] <= 1.0)
        & (ndc[:, 2] >= -1.0)
        & (ndc[:, 2] <= 1.0)
    )
    if not np.any(inside):
        return None

    ndc = ndc[inside]
    min_ndc = ndc.min(axis=0)
    max_ndc = ndc.max(axis=0)

    px_min = ((min_ndc[0] + 1.0) * 0.5) * width
    px_max = ((max_ndc[0] + 1.0) * 0.5) * width
    py_min = ((min_ndc[1] + 1.0) * 0.5) * height
    py_max = ((max_ndc[1] + 1.0) * 0.5) * height

    pad_x = (px_max - px_min) * padding_ratio
    pad_y = (py_max - py_min) * padding_ratio
    px_min -= pad_x
    px_max += pad_x
    py_min -= pad_y
    py_max += pad_y

    x0 = max(0.0, min(float(width), px_min))
    x1 = max(0.0, min(float(width), px_max))
    y0 = max(0.0, min(float(height), py_min))
    y1 = max(0.0, min(float(height), py_max))

    x0_i = int(np.floor(x0))
    y0_i = int(np.floor(y0))
    x1_i = int(np.ceil(x1))
    y1_i = int(np.ceil(y1))

    if x0_i >= x1_i or y0_i >= y1_i:
        return None

    pixel_area = float((x1_i - x0_i) * (y1_i - y0_i))
    return [x0_i, y0_i, x1_i, y1_i], pixel_area


def _load_tile_points(
    storage: TileStorage,
    coords: List[Tuple[int, int, int]],
    max_points: int,
    seed: int,
) -> Dict[Tuple[int, int, int], np.ndarray]:
    rng = default_rng(seed)
    tile_points: Dict[Tuple[int, int, int], np.ndarray] = {}
    for coord in tqdm(coords, desc="Loading tile points"):
        tile = storage.load_tile(coord)
        if tile is None:
            continue
        means = tile["means"].astype(np.float64, copy=False)
        if max_points > 0 and means.shape[0] > max_points:
            indices = rng.choice(means.shape[0], size=max_points, replace=False)
            means = means[indices]
        tile_points[coord] = np.ascontiguousarray(means)
    return tile_points


def _create_camera_package(cam_info: Dict, view: Dict, zfar: float) -> Dict:
    R_wc = qvec2rotmat(view["qvec"])  # world <- camera
    R_cw = R_wc.T
    T_wc = view["tvec"]

    world_view_np = getWorld2View2(R_cw, T_wc)
    world_view = world_view_np.astype(np.float32).T

    projection = getProjectionMatrix(
        znear=0.01,
        zfar=zfar,
        fovX=cam_info["fovx"],
        fovY=cam_info["fovy"],
    )
    projection_np = projection.cpu().numpy().astype(np.float32).T

    full_proj = world_view @ projection_np

    return {
        "world_view": world_view,
        "projection": projection_np,
        "full_proj": full_proj,
        "width": cam_info["width"],
        "height": cam_info["height"],
    }


def build_visibility(
    storage: TileStorage,
    views: List[Dict],
    camera_params: Dict[int, Dict],
    padding_ratio: float,
    min_pixel_area: float,
    zfar: float,
    max_points_per_tile: int,
    point_sample_seed: int,
) -> Dict:
    tile_coords = storage.list_all_tiles()
    tile_infos = {coord: storage.get_tile_info(coord) for coord in tile_coords}
    valid_coords = [coord for coord, info in tile_infos.items() if info is not None]
    tile_points = _load_tile_points(storage, valid_coords, max_points_per_tile, point_sample_seed)

    tiles_result: Dict[str, Dict] = {}
    cameras_result: Dict[int, Dict] = {}

    for view in tqdm(views, desc="Processing cameras"):
        cam_info = camera_params[view["cam_id"]]
        cam_pack = _create_camera_package(cam_info, view, zfar)
        camera_entry = {
            "index": view["index"],
            "image_name": view["image_name"],
            "cam_id": view["cam_id"],
            "tiles": [],
        }

        for coord in valid_coords:
            info = tile_infos[coord]
            points = tile_points.get(coord)
            if info is None or points is None:
                continue
            projection = _project_points(
                cam_pack["full_proj"],
                cam_pack["width"],
                cam_pack["height"],
                points,
                padding_ratio,
            )
            if projection is None:
                continue
            bbox_px, pixel_area = projection
            if pixel_area < min_pixel_area:
                continue

            tile_id = f"{coord[0]}_{coord[1]}_{coord[2]}"
            tile_entry = tiles_result.setdefault(
                tile_id,
                {
                    "coord": list(coord),
                    "num_gaussians": int(info["num_gaussians"]),
                    "cameras": [],
                },
            )

            cam_payload = {
                "camera_index": view["index"],
                "image_name": view["image_name"],
                "bbox": [int(v) for v in bbox_px],
                "pixel_area": pixel_area,
                "width": cam_pack["width"],
                "height": cam_pack["height"],
            }
            tile_entry["cameras"].append(cam_payload)
            camera_entry["tiles"].append(
                {
                    "tile_id": tile_id,
                    "coord": list(coord),
                    "bbox": cam_payload["bbox"],
                    "pixel_area": pixel_area,
                    "num_gaussians": int(info["num_gaussians"]),
                    "width": cam_pack["width"],
                    "height": cam_pack["height"],
                }
            )

        cameras_result[view["index"]] = camera_entry

    return {
        "tiles": tiles_result,
        "cameras": cameras_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute tile visibility for a COLMAP scene",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tiles", required=True, help="Tiled scene directory")
    parser.add_argument("--colmap", required=True, help="COLMAP sparse/0 directory")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <tiles>/tile_visibility.json)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Padding ratio applied to tile bounding boxes before projection",
    )
    parser.add_argument(
        "--min-pixel-area",
        type=float,
        default=64.0,
        help="Minimum projected pixel area to keep a tile-camera pair",
    )
    parser.add_argument(
        "--zfar",
        type=float,
        default=10000.0,
        help="Far plane distance used for projection matrix",
    )
    parser.add_argument(
        "--max-points-per-tile",
        type=int,
        default=0,
        help="Limit the number of Gaussian centers projected per tile (0 means use all)",
    )
    parser.add_argument(
        "--point-sample-seed",
        type=int,
        default=0,
        help="Random seed used when subsampling tile points",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tile_dir = Path(args.tiles).resolve()
    colmap_dir = Path(args.colmap).resolve()
    if args.output is None:
        output_path = tile_dir / "tile_visibility.json"
    else:
        output_path = Path(args.output).resolve()

    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile directory not found: {tile_dir}")
    if not colmap_dir.exists():
        raise FileNotFoundError(f"COLMAP directory not found: {colmap_dir}")

    print("Loading tiles metadata...")
    storage = TileStorage(str(tile_dir))

    print("Loading COLMAP camera parameters...")
    camera_params = _load_colmap_cameras(colmap_dir)
    views = _load_colmap_images(colmap_dir)
    print(f"  Parsed {len(camera_params)} camera intrinsics")
    print(f"  Parsed {len(views)} camera poses")

    print("Computing visibility matrix...")
    visibility = build_visibility(
        storage=storage,
        views=views,
        camera_params=camera_params,
        padding_ratio=args.padding,
        min_pixel_area=args.min_pixel_area,
        zfar=args.zfar,
        max_points_per_tile=args.max_points_per_tile,
        point_sample_seed=args.point_sample_seed,
    )
    storage.close()

    # Append summary statistics
    num_tile_pairs = sum(len(v["cameras"]) for v in visibility["tiles"].values())
    num_camera_pairs = sum(len(v["tiles"]) for v in visibility["cameras"].values())
    summary = {
        "padding_ratio": args.padding,
        "min_pixel_area": args.min_pixel_area,
        "zfar": args.zfar,
        "num_tiles": len(visibility["tiles"]),
        "num_cameras": len(visibility["cameras"]),
        "tile_camera_pairs": num_tile_pairs,
        "camera_tile_pairs": num_camera_pairs,
        "source_tiles": str(tile_dir),
        "source_colmap": str(colmap_dir),
    }

    output = {"summary": summary, **visibility}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Visibility metadata written to {output_path}")


if __name__ == "__main__":
    main()

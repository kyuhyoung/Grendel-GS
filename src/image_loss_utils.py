"""Utilities for COLMAP-driven image supervision of Gaussian tiles."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent.parent

# Ensure Grendel modules are importable when this utility is used outside the package root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "Grendel-GS"))

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.colmap_loader import qvec2rotmat
from utils.graphics_utils import getProjectionMatrix, getWorld2View2

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from scene.gaussian_model import GaussianModel


def load_colmap_camera(colmap_dir: Path) -> Dict[str, Any]:
    """Return the first supported COLMAP camera intrinsics."""

    cameras_file = colmap_dir / "cameras.txt"
    if not cameras_file.exists():
        raise FileNotFoundError(f"cameras.txt not found under {colmap_dir}")

    with open(cameras_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            if model == "PINHOLE":
                fx, fy, cx, cy = params
            elif model == "SIMPLE_PINHOLE":
                f, cx, cy = params
                fx = fy = f
            else:
                continue
            fovx = 2 * math.atan(width / (2 * fx))
            fovy = 2 * math.atan(height / (2 * fy))
            return {
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "fovx": fovx,
                "fovy": fovy,
            }

    raise RuntimeError(f"No supported camera models found in {cameras_file}")


def load_colmap_pose(colmap_dir: Path, image_idx: int) -> Dict[str, Any]:
    """Fetch pose + metadata for the image at ``image_idx`` from COLMAP."""

    images_file = colmap_dir / "images.txt"
    if not images_file.exists():
        raise FileNotFoundError(f"images.txt not found under {colmap_dir}")

    entries: List[List[str]] = []
    with open(images_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 10:
                continue
            name_token = parts[-1]
            if "." not in name_token:
                continue
            entries.append(parts)

    if not entries:
        raise RuntimeError(f"No image entries found in {images_file}")
    if image_idx < 0 or image_idx >= len(entries):
        raise RuntimeError(f"Image index {image_idx} not found in {images_file}")

    parts = entries[image_idx]
    qw, qx, qy, qz = map(float, parts[1:5])
    tx, ty, tz = map(float, parts[5:8])
    cam_id = int(parts[8])
    name = parts[9]
    return {
        "quat": np.array([qw, qx, qy, qz], dtype=np.float32),
        "translation": np.array([tx, ty, tz], dtype=np.float32),
        "cam_id": cam_id,
        "name": name,
    }


def build_camera_from_colmap(
    cam_params: Dict[str, Any],
    pose_params: Dict[str, Any],
    device: torch.device,
    resolution_scale: float,
    *,
    znear: float = 0.01,
    zfar: float = 10000.0,
) -> Dict[str, Any]:
    """Construct rasterizer matrices for a COLMAP pose."""

    width = max(1, int(round(cam_params["width"] * resolution_scale)))
    height = max(1, int(round(cam_params["height"] * resolution_scale)))

    R_wc = qvec2rotmat(pose_params["quat"])
    R_cw = R_wc.T
    T_wc = pose_params["translation"]

    world_view_np = getWorld2View2(R_cw, T_wc)
    world_view_transform = torch.tensor(world_view_np, dtype=torch.float32, device=device).transpose(0, 1)

    projection = getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam_params["fovx"], fovY=cam_params["fovy"])
    projection_matrix = projection.to(device=device, dtype=torch.float32).transpose(0, 1)

    full_proj = world_view_transform @ projection_matrix
    camera_center = torch.linalg.inv(world_view_transform)[3, :3]

    return {
        "image_height": height,
        "image_width": width,
        "tanfovx": float(np.tan(cam_params["fovx"] * 0.5)),
        "tanfovy": float(np.tan(cam_params["fovy"] * 0.5)),
        "world_view_transform": world_view_transform,
        "projection_matrix": projection_matrix,
        "full_proj_transform": full_proj,
        "camera_center": camera_center,
        "original_resolution": (cam_params["width"], cam_params["height"]),
    }


def split_camera_vertical(
    camera,  # Can be Dict or Namespace
    split_index: int,
    total_splits: int = 2,
    overlap_pixels: int = 0,
):
    """Split camera vertically into sub-regions for memory-efficient rendering.
    
    Args:
        camera: Original camera dict or Namespace from build_camera_from_colmap.
        split_index: Index of the sub-region (0 = top, 1 = bottom for 2-way split).
        total_splits: Total number of vertical splits (default: 2).
        overlap_pixels: Number of pixels to overlap between splits (adds to top and bottom).
    
    Returns:
        New camera dict or Namespace with adjusted projection and resolution for the sub-region.
    """
    if split_index >= total_splits:
        raise ValueError(f"split_index ({split_index}) must be < total_splits ({total_splits})")
    
    # Support both dict and Namespace
    is_namespace = not isinstance(camera, dict)
    
    def get_attr(obj, key):
        """Get attribute from dict or Namespace"""
        if isinstance(obj, dict):
            return obj[key]
        else:
            return getattr(obj, key)
    
    original_height = get_attr(camera, "image_height")
    original_width = get_attr(camera, "image_width")
    
    # Calculate base sub-region height (without overlap)
    base_sub_height = original_height // total_splits
    
    # Calculate vertical offset for the base region
    base_y_offset = split_index * base_sub_height
    
    # Adjust for the last split to cover remaining pixels
    if split_index == total_splits - 1:
        base_sub_height = original_height - base_y_offset
        
    # Calculate actual render region with overlap
    # We add overlap to top and bottom, but clamp to image bounds
    start_y = max(0, base_y_offset - overlap_pixels)
    end_y = min(original_height, base_y_offset + base_sub_height + overlap_pixels)
    
    actual_height = end_y - start_y
    
    # Calculate the effective y_offset for the projection matrix
    # This is the offset of the top of the rendered region relative to the full image
    y_offset = start_y
    
    # Calculate sub-region height
    sub_height = actual_height
    
    # Calculate new principal point offset in NDC space
    # NDC ranges from -1 to 1, and we need to shift the viewing frustum
    # Original cy is for full image, new cy should be for sub-region
    
    # tanfovy relates to the half-height FOV
    # For a sub-region starting at y_offset with height sub_height:
    # We need to compute new NDC bounds
    
    # Scale factor for the sub-region
    scale_y = original_height / sub_height
    
    # Build new projection matrix with off-center frustum
    # Start from original projection matrix
    new_projection = get_attr(camera, "projection_matrix").clone()
    
    # Modify projection matrix for vertical offset
    # The projection matrix from build_camera_from_colmap is TRANSPOSED.
    # Standard OpenGL P:
    # [ x_scale  0        0        0 ]
    # [ 0        y_scale  y_off    0 ]
    # [ 0        0        z_A      z_B ]
    # [ 0        0        -1       0 ]
    #
    # Transposed P_T:
    # [ x_scale  0        0        0 ]
    # [ 0        y_scale  0        0 ]
    # [ 0        y_off    z_A      -1 ]
    # [ 0        0        z_B      0 ]
    
    # P[1,1] controls vertical scaling (FOV) - Diagonal, so same for T
    # P[2,1] controls vertical translation (principal point offset) in Transposed matrix
    
    # Scale the vertical FOV (Zoom in)
    # P[1,1] = 1 / tan(fov/2). If we zoom in by scale_y, P[1,1] increases.
    new_projection[1, 1] = get_attr(camera, "projection_matrix")[1, 1] * scale_y
    
    # Shift the vertical center (Principal Point adjustment)
    # We need to modify P_T[2, 1] (which corresponds to P[1, 2])
    # Formula: P'[1,2] = S * P[1,2] + (S - 1) - S * (2 * y_offset / H)
    # This correctly maps the new crop to the full NDC range [-1, 1]
    # Verified:
    # Top Crop (y=0, S=2): P'[1,2] = 2*0 + 1 - 0 = 1. (Maps y_ndc=1->1, y_ndc=0->-1)
    # Bottom Crop (y=H/2, S=2): P'[1,2] = 2*0 + 1 - 2 = -1. (Maps y_ndc=0->1, y_ndc=-1->-1)
    
    p_1_2 = get_attr(camera, "projection_matrix")[2, 1]
    new_p_1_2 = scale_y * p_1_2 + (scale_y - 1.0) - scale_y * (2.0 * y_offset / original_height)
    
    new_projection[2, 1] = new_p_1_2
    
    # Recompute full projection transform
    new_full_proj = get_attr(camera, "world_view_transform") @ new_projection
    
    # Update tanfovy
    # tan(fov/2) = 1 / P[1,1].
    # Since P[1,1] was multiplied by scale_y, tan(fov/2) is divided by scale_y.
    new_tanfovy = get_attr(camera, "tanfovy") / scale_y

    # Build result (dict or Namespace based on input)
    result_dict = {
        "image_height": sub_height,
        "image_width": original_width,
        "tanfovx": get_attr(camera, "tanfovx"),
        "tanfovy": new_tanfovy,  # Corrected: Divide by scale
        "world_view_transform": get_attr(camera, "world_view_transform"),
        "projection_matrix": new_projection,
        "full_proj_transform": new_full_proj,
        "camera_center": get_attr(camera, "camera_center"),
        "original_resolution": get_attr(camera, "original_resolution") if hasattr(camera, "original_resolution") or "original_resolution" in camera else (original_width, original_height),
        "split_info": {
            "index": split_index,
            "total": total_splits,
            "y_offset": y_offset,
        }
    }
    
    # Copy other attributes if Namespace
    if is_namespace:
        # Create new Namespace or copy?
        # We can't easily copy Namespace and update, so we create dict and convert
        # But we need to preserve other attributes like uid, image_name etc if they exist
        # The caller should handle copying extra attributes
        return result_dict
    else:
        return result_dict


def load_reference_image(
    images_root: Path,
    image_name: str,
    width: int,
    height: int,
    device: torch.device,
) -> torch.Tensor:
    """Load and optionally resize the RGB reference frame."""

    image_path = images_root / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Reference image '{image_name}' not found under {images_root}")

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)
    return tensor


def _make_cuda_args() -> Dict[str, Any]:
    return {
        "mode": "test",
        "world_size": "1",
        "global_rank": "0",
        "local_rank": "0",
        "mp_world_size": "1",
        "mp_rank": "0",
        "log_folder": ".",
        "log_interval": "1",
        "iteration": "0",
        "zhx_debug": "False",
        "zhx_time": "False",
        "avoid_pixel_all2all": False,
        "stats_collector": {},
    }


def render_gaussian_model(
    model: "GaussianModel",
    camera: Dict[str, Any],
    bg_color: torch.Tensor,
    *,
    scaling_modifier: float = 1.0,
    region: Optional[Tuple[int, int, int, int]] = None,
    return_region_only: bool = False,
) -> torch.Tensor:
    """Rasterize the provided Gaussian model from a prebuilt camera.

    Args:
        model: Gaussian model to render.
        camera: Camera dictionary from ``prepare_image_loss``.
        bg_color: Background color tensor.
        scaling_modifier: Optional scaling modifier passed to the rasterizer.
        region: Optional pixel-space bounding box (x0, y0, x1, y1) within the
            camera's scaled resolution. When provided, the rasterizer only processes
            the tiles that overlap this region.
        return_region_only: When ``True`` and ``region`` is provided, only the
            rendered crop is returned instead of the full frame.
    """

    settings = GaussianRasterizationSettings(
        image_height=int(camera["image_height"]),
        image_width=int(camera["image_width"]),
        tanfovx=float(camera["tanfovx"]),
        tanfovy=float(camera["tanfovy"]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera["world_view_transform"],
        projmatrix=camera["full_proj_transform"],
        sh_degree=model.active_sh_degree,
        campos=camera["camera_center"],
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=settings)
    cuda_args = _make_cuda_args()

    means2D, rgb, conic_opacity, radii, depths = rasterizer.preprocess_gaussians(
        means3D=model.get_xyz,
        scales=model.get_scaling,
        rotations=model.get_rotation,
        shs=model.get_features,
        opacities=model.get_opacity,
        cuda_args=cuda_args,
    )

    width = int(camera["image_width"])
    height = int(camera["image_height"])

    clipped_region: Optional[Tuple[int, int, int, int]] = None
    if region is not None:
        x0, y0, x1, y1 = region
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(x0 + 1, min(width, x1))
        y1 = max(y0 + 1, min(height, y1))
        if x1 > x0 and y1 > y0:
            clipped_region = (x0, y0, x1, y1)

    tile_x = (width + 15) // 16
    tile_y = (height + 15) // 16
    if clipped_region is None:
        compute_locally = torch.ones((tile_y, tile_x), dtype=torch.bool, device=bg_color.device)
    else:
        compute_locally = torch.zeros((tile_y, tile_x), dtype=torch.bool, device=bg_color.device)
        x0, y0, x1, y1 = clipped_region
        start_tx = max(0, x0 // 16)
        end_tx = min(tile_x, (x1 + 15) // 16)
        start_ty = max(0, y0 // 16)
        end_ty = min(tile_y, (y1 + 15) // 16)
        compute_locally[start_ty:end_ty, start_tx:end_tx] = True
        if not compute_locally.any():
            compute_locally.fill_(True)
            clipped_region = None

    results = rasterizer.render_gaussians(
        means2D=means2D,
        conic_opacity=conic_opacity,
        rgb=rgb,
        depths=depths,
        radii=radii,
        compute_locally=compute_locally,
        extended_compute_locally=compute_locally,
        cuda_args=cuda_args,
    )

    if isinstance(results, tuple):
        rendered = results[0]
    else:
        rendered = results

    if clipped_region is not None and return_region_only:
        x0, y0, x1, y1 = clipped_region
        return rendered[:, y0:y1, x0:x1]

    return rendered


def prepare_image_loss(
    *,
    device: torch.device,
    colmap_dir: Path,
    images_dir: Path,
    image_idx: int,
    resolution_scale: float,
    znear: float = 0.01,
    zfar: float = 10000.0,
) -> Dict[str, Any]:
    """Prepare camera, target tensor, and reusable buffers for image supervision."""

    if not colmap_dir.exists():
        raise FileNotFoundError(f"COLMAP directory not found: {colmap_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    cam_params = load_colmap_camera(colmap_dir)
    pose_params = load_colmap_pose(colmap_dir, image_idx)
    camera = build_camera_from_colmap(
        cam_params,
        pose_params,
        device=device,
        resolution_scale=resolution_scale,
        znear=znear,
        zfar=zfar,
    )
    target = load_reference_image(images_dir, pose_params["name"], camera["image_width"], camera["image_height"], device)

    return {
        "camera": camera,
        "target": target,
        "target_cpu": target.detach().cpu(),
        "bg_color": torch.ones(3, device=device),
        "image_name": pose_params["name"],
    }


def save_image_tensor(image: torch.Tensor, path: Path) -> None:
    """Persist a rendered RGB tensor to disk."""

    arr = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_abs_diff_image(render: torch.Tensor, target: torch.Tensor, path: Path) -> None:
    """Write an absolute-difference heatmap between two RGB tensors."""

    render_np = render.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    target_np = target.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    diff = np.abs(render_np - target_np)
    max_val = float(diff.max())
    if max_val > 0.0:
        diff = diff / max_val
    img = Image.fromarray((diff * 255.0).astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def create_window(window_size, channel):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


class ColmapLoader:
    """Cached loader for COLMAP metadata."""
    
    def __init__(self, colmap_dir: Path):
        self.colmap_dir = colmap_dir
        self._cameras = {}
        self._images = {}
        self._load_cameras()
        self._load_images()

    def _load_cameras(self):
        cameras_file = self.colmap_dir / "cameras.txt"
        if not cameras_file.exists():
            raise FileNotFoundError(f"cameras.txt not found under {self.colmap_dir}")

        with open(cameras_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 4:
                    continue
                cam_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]
                
                if model == "PINHOLE":
                    fx, fy, cx, cy = params
                elif model == "SIMPLE_PINHOLE":
                    f, cx, cy = params
                    fx = fy = f
                elif model == "RADIAL":
                    f, cx, cy = params[:3]  # k1, k2 ignored
                    fx = fy = f
                elif model == "SIMPLE_RADIAL":
                    f, cx, cy = params[:3]  # k ignored
                    fx = fy = f
                else:
                    print(f"[Warning] Unsupported camera model: {model}, skipping")
                    continue
                    
                fovx = 2 * math.atan(width / (2 * fx))
                fovy = 2 * math.atan(height / (2 * fy))
                
                self._cameras[cam_id] = {
                    "width": width,
                    "height": height,
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "fovx": fovx,
                    "fovy": fovy,
                }

    def _load_images(self):
        images_file = self.colmap_dir / "images.txt"
        if not images_file.exists():
            raise FileNotFoundError(f"images.txt not found under {self.colmap_dir}")

        entries = []
        with open(images_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 10:
                    continue
                name_token = parts[-1]
                if "." not in name_token:
                    continue
                entries.append(parts)
        
        for idx, parts in enumerate(entries):
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            name = parts[9]
            self._images[idx] = {
                "quat": np.array([qw, qx, qy, qz], dtype=np.float32),
                "translation": np.array([tx, ty, tz], dtype=np.float32),
                "cam_id": cam_id,
                "name": name,
            }

    def get_camera(self, cam_id: int) -> Dict[str, Any]:
        if cam_id not in self._cameras:
             # Fallback to first camera if specific ID not found (legacy behavior)
             if self._cameras:
                 return next(iter(self._cameras.values()))
             raise KeyError(f"Camera {cam_id} not found")
        return self._cameras[cam_id]

    def get_pose(self, image_idx: int) -> Dict[str, Any]:
        if image_idx not in self._images:
            raise KeyError(f"Image index {image_idx} not found")
        return self._images[image_idx]

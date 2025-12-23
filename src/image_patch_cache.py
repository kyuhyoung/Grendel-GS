"""Visibility-aware image patch cache for tile supervision.

This module provides a small utility that crops RGB patches from large COLMAP
images using the projected bounding boxes stored in ``tile_visibility.json``.
The cache keeps resized full images (at a given resolution scale) and
optionally individual patches so repeated tile/camera lookups avoid disk IO.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from collections import OrderedDict
import math
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
import atexit
import warnings

import numpy as np
import torch
from PIL import Image

# Suppress shared_memory cleanup warnings on abnormal exit
# These warnings appear when shared memory is already cleaned up by OS
# but resource_tracker still tries to verify cleanup
warnings.filterwarnings('ignore', message='resource_tracker: .*shared_memory.*', category=UserWarning)

from .visibility_utils import TileObservation, TileVisibility, VisibilityDataset


@dataclass
class PatchResult:
    """Container describing a cached patch."""

    image_name: str
    camera_index: int
    tile_ids: Tuple[str, ...]
    bbox: Tuple[int, int, int, int]
    scaled_bbox: Tuple[int, int, int, int]
    scale: float
    tensor: torch.Tensor


class ImagePatchCache:
    """Cache that serves cropped/resized patches for tile-camera pairs.

    Parameters
    ----------
    images_root:
        Path containing the original COLMAP RGB images.
    dataset:
        Visibility metadata loaded via :func:`visibility_utils.load_visibility_metadata`.
    max_scaled_images:
        Maximum number of resized full images to keep resident. When exceeded,
        the least-recently-used item is evicted.
    default_scale:
        Resolution scale applied when callers do not pass ``resolution_scale``.
    """

    def __init__(
        self,
        images_root: Path,
        dataset: VisibilityDataset,
        *,
        max_scaled_images: int = 8,
        default_scale: float = 1.0,
    ) -> None:
        self.images_root = Path(images_root)
        self.dataset = dataset
        self.max_scaled_images = max(1, max_scaled_images)
        self.default_scale = float(default_scale)
        self._scaled_cache: "OrderedDict[Tuple[str, float], torch.Tensor]" = OrderedDict()
        self._patch_cache: "OrderedDict[Tuple[str, Tuple[int, int, int, int], float], torch.Tensor]" = OrderedDict()
        self._dimension_cache: Dict[str, Tuple[int, int]] = {}
        self.stats = {
            "image_loads": 0,
            "scaled_image_builds": 0,
            "scaled_image_hits": 0,
            "patch_builds": 0,
            "patch_hits": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_patch(
        self,
        tile_id: str,
        camera_index: int,
        *,
        resolution_scale: Optional[float] = None,
        padding_px: int = 0,
    ) -> PatchResult:
        obs = self._find_observation(tile_id, camera_index)
        bbox = self._expand_bbox(obs, padding_px)
        return self._build_patch([tile_id], obs, bbox, resolution_scale)

    def get_union_patch(
        self,
        tile_ids: Sequence[str],
        camera_index: int,
        *,
        resolution_scale: Optional[float] = None,
        padding_px: int = 0,
    ) -> PatchResult:
        if not tile_ids:
            raise ValueError("tile_ids must be non-empty")
        observations = [self._find_observation(tid, camera_index) for tid in tile_ids]
        bbox = self._union_bbox(observations, padding_px)
        # All observations share the same image/camera.
        return self._build_patch(list(tile_ids), observations[0], bbox, resolution_scale)

    def get_crop_image(
        self,
        image_name: str,
        crop_box: Tuple[int, int, int, int],
        resolution_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Retrieve a cropped region of the image, scaled to the requested resolution.

        Args:
            image_name: Name of the image file.
            crop_box: (left, top, right, bottom) in ORIGINAL image coordinates.
            resolution_scale: Scale factor to apply AFTER cropping.
        """
        scale = float(resolution_scale) if resolution_scale is not None else self.default_scale

        # Use cached full image to avoid disk IO
        # This leverages the existing LRU cache for scaled images
        full_image = self._get_scaled_image(image_name, scale)
        
        # Map crop box to scaled coordinates
        # crop_box is (x0, y0, x1, y1) in original resolution
        x0, y0, x1, y1 = crop_box
        crop_w = x1 - x0
        crop_h = y1 - y0
        
        # Calculate scaled coordinates matching the rounding logic in build_camera_from_colmap
        sx0 = int(round(x0 * scale))
        sy0 = int(round(y0 * scale))
        
        target_w = max(1, int(round(crop_w * scale)))
        target_h = max(1, int(round(crop_h * scale)))
        
        sx1 = sx0 + target_w
        sy1 = sy0 + target_h
        
        # Clamp to image bounds
        _, H, W = full_image.shape
        sx0 = max(0, min(W, sx0))
        sy0 = max(0, min(H, sy0))
        sx1 = max(0, min(W, sx1))
        sy1 = max(0, min(H, sy1))
        
        crop = full_image[:, sy0:sy1, sx0:sx1].contiguous()
        if crop.dtype == torch.uint8:
            crop = crop.float() / 255.0
        return crop

    def get_full_image(
        self,
        image_name: str,
        resolution_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Retrieve the full image, scaled to the requested resolution.

        Args:
            image_name: Name of the image file.
            resolution_scale: Scale factor to apply to the full image.
        """
        img = self._get_scaled_image(image_name, resolution_scale)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        return img

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_observation(self, tile_id: str, camera_index: int) -> TileObservation:
        tile: Optional[TileVisibility] = self.dataset.tiles.get(tile_id)
        if tile is None:
            raise KeyError(f"Tile {tile_id} not present in visibility metadata")
        for obs in tile.cameras:
            if obs.camera_index == camera_index:
                if not obs.image_name:
                    raise KeyError(f"Tile {tile_id} has empty image name metadata")
                return obs
        raise KeyError(f"Tile {tile_id} is not visible to camera {camera_index}")

    def _expand_bbox(self, obs: TileObservation, padding_px: int) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = obs.bbox
        if padding_px > 0:
            width, height = self._observation_dimensions(obs)
            x0 = max(0, x0 - padding_px)
            y0 = max(0, y0 - padding_px)
            x1 = min(width, x1 + padding_px)
            y1 = min(height, y1 + padding_px)
        return x0, y0, x1, y1

    def _union_bbox(
        self,
        observations: Sequence[TileObservation],
        padding_px: int,
    ) -> Tuple[int, int, int, int]:
        mins = [self._expand_bbox(obs, padding_px) for obs in observations]
        x0 = min(b[0] for b in mins)
        y0 = min(b[1] for b in mins)
        x1 = max(b[2] for b in mins)
        y1 = max(b[3] for b in mins)
        first = observations[0]
        width, height = self._observation_dimensions(first)
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        return x0, y0, x1, y1

    def _build_patch(
        self,
        tile_ids: Sequence[str],
        obs: TileObservation,
        bbox: Tuple[int, int, int, int],
        resolution_scale: Optional[float],
    ) -> PatchResult:
        scale = float(resolution_scale) if resolution_scale is not None else self.default_scale
        scaled_image = self._get_scaled_image(obs.image_name, scale, obs)
        scaled_bbox = self._scale_bbox(bbox, scale, obs)
        key = (obs.image_name, scaled_bbox, scale)
        patch_tensor = self._patch_cache.get(key)
        if patch_tensor is None:
            x0, y0, x1, y1 = scaled_bbox
            patch_tensor = scaled_image[:, y0:y1, x0:x1].contiguous().clone()
            if patch_tensor.dtype == torch.uint8:
                patch_tensor = patch_tensor.float() / 255.0
            self._patch_cache[key] = patch_tensor
            self.stats["patch_builds"] += 1
        else:
            self.stats["patch_hits"] += 1
        return PatchResult(
            image_name=obs.image_name,
            camera_index=obs.camera_index,
            tile_ids=tuple(tile_ids),
            bbox=bbox,
            scaled_bbox=scaled_bbox,
            scale=scale,
            tensor=patch_tensor.clone(),
        )

    def _get_scaled_image(
        self,
        image_name: str,
        scale: float,
        obs: Optional[TileObservation] = None,
    ) -> torch.Tensor:
        key = (image_name, scale)
        tensor = self._scaled_cache.get(key)
        if tensor is not None:
            self._scaled_cache.move_to_end(key)
            self.stats["scaled_image_hits"] += 1
            return tensor

        image_t = self._load_image_tensor(image_name, scale, obs)
        self._scaled_cache[key] = image_t
        self.stats["scaled_image_builds"] += 1
        if len(self._scaled_cache) > self.max_scaled_images:
            self._scaled_cache.popitem(last=False)
        return image_t

    def _load_image_tensor(
        self,
        image_name: str,
        scale: float,
        obs: Optional[TileObservation] = None,
    ) -> torch.Tensor:
        image_path = self.images_root / image_name
        if not image_path.exists():
            # Try absolute path if image_name is absolute (though unlikely in this setup)
            if Path(image_name).exists():
                image_path = Path(image_name)
            else:
                raise FileNotFoundError(f"Image '{image_name}' not found under {self.images_root} (Absolute: {image_path.absolute()})")
        
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size
                target_w, target_h = self._scaled_resolution(width, height, scale)
                if (target_w, target_h) != (width, height):
                    img = img.resize((target_w, target_h), Image.LANCZOS)
                # Store as uint8 to save memory (4x reduction vs float32)
                # Use copy() to ensure the array is writable and owns its memory, avoiding PyTorch warnings
                arr = np.asarray(img, dtype=np.uint8).copy()
            self.stats["image_loads"] += 1
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            return tensor
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {image_path.absolute()}: {e}") from e

    def _observation_dimensions(self, obs: TileObservation) -> Tuple[int, int]:
        if obs.image_width and obs.image_height:
            return obs.image_width, obs.image_height
        cached = self._dimension_cache.get(obs.image_name)
        if cached is None:
            image_path = self.images_root / obs.image_name
            with Image.open(image_path) as img:
                cached = img.size
            self._dimension_cache[obs.image_name] = cached
        return cached

    @staticmethod
    def _scaled_resolution(width: int, height: int, scale: float) -> Tuple[int, int]:
        if scale <= 0:
            raise ValueError("scale must be positive")
        target_w = max(1, int(round(width * scale)))
        target_h = max(1, int(round(height * scale)))
        return target_w, target_h

    def _scale_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        scale: float,
        obs: TileObservation,
    ) -> Tuple[int, int, int, int]:
        width, height = self._observation_dimensions(obs)
        scaled_w, scaled_h = self._scaled_resolution(width, height, scale)
        x0, y0, x1, y1 = bbox
        sx0 = max(0, min(scaled_w, int(math.floor(x0 * scale))))
        sy0 = max(0, min(scaled_h, int(math.floor(y0 * scale))))
        sx1 = max(0, min(scaled_w, int(math.ceil(x1 * scale))))
        sy1 = max(0, min(scaled_h, int(math.ceil(y1 * scale))))
        if sx0 >= sx1 or sy0 >= sy1:
            raise ValueError("Scaled bbox is empty after clamping")
        return sx0, sy0, sx1, sy1


@dataclass
class SharedMemoryInfo:
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    image_name: str
    scale: float

class SharedMemoryImageCache(ImagePatchCache):
    """
    Extension of ImagePatchCache that supports storing images in Shared Memory.
    Designed for distributed training where Rank 0 loads and others read.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shm_blocks: "OrderedDict[str, shared_memory.SharedMemory]" = OrderedDict()
        self._shm_metadata: Dict[str, Tuple[Tuple[int, ...], torch.dtype]] = {}
        self._created_shm_names: set[str] = set()
        # Register cleanup to ensure we don't leak SHM on exit
        atexit.register(self.cleanup)

    def cleanup(self):
        """Unlink all allocated shared memory blocks."""
        for name, shm in list(self._shm_blocks.items()):
            try:
                # Unregister from resource tracker before cleanup
                # This prevents warnings about leaked shared_memory objects
                try:
                    resource_tracker.unregister(shm._name, "shared_memory")
                except Exception:
                    pass
                
                shm.close()
            except Exception as e:
                # Already closed, that's fine
                pass
            
            # Unlink only if we created it
            if name in self._created_shm_names:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    # Already unlinked, that's fine
                    pass
                except Exception as e:
                    # Log but continue cleanup
                    pass
        self._shm_blocks.clear()
        self._created_shm_names.clear()
        
    def unlink_all(self):
        """Force unlink of all tracked blocks (Call this from Rank 0)."""
        for name, shm in self._shm_blocks.items():
            try:
                shm.unlink()
            except Exception:
                pass
        self._shm_blocks.clear()
        self._created_shm_names.clear()

    def put_image_to_shm(self, image_name: str, resolution_scale: float = 1.0) -> SharedMemoryInfo:
        """
        Load image, write to Shared Memory, and return metadata.
        Only to be called by the writer process (Rank 0).
        """
        # Use a unique name based on image and scale
        shm_name = f"img_{hash(image_name)}_{resolution_scale}"
        
        # Check if we already have it open and have metadata
        if shm_name in self._shm_blocks and shm_name in self._shm_metadata:
             # Move to end (MRU)
             self._shm_blocks.move_to_end(shm_name)
             shape, dtype = self._shm_metadata[shm_name]
             return SharedMemoryInfo(
                name=shm_name,
                shape=shape,
                dtype=dtype,
                image_name=image_name,
                scale=resolution_scale
            )

        # Evict if full
        while len(self._shm_blocks) >= self.max_scaled_images:
            oldest_name, oldest_shm = self._shm_blocks.popitem(last=False)
            if oldest_name in self._created_shm_names:
                try:
                    oldest_shm.unlink()
                except Exception:
                    pass
            oldest_shm.close()
            if oldest_name in self._shm_metadata:
                del self._shm_metadata[oldest_name]
            if oldest_name in self._created_shm_names:
                self._created_shm_names.remove(oldest_name)

        # 1. Get the uint8 tensor
        # We use _load_image_tensor to avoid caching in _scaled_cache (RAM), 
        # as we are about to cache it in Shared Memory.
        tensor = self._load_image_tensor(image_name, resolution_scale)
        
        # 2. Create Shared Memory
        # Check if we already have it open (but missing metadata for some reason)
        if shm_name in self._shm_blocks:
             shm = self._shm_blocks[shm_name]
        else:
            size = tensor.nelement() * tensor.element_size()
            try:
                # Try to create
                shm = shared_memory.SharedMemory(create=True, size=size, name=shm_name)
                self._created_shm_names.add(shm_name)
            except FileExistsError:
                # If exists, attach and overwrite (or assume it's good)
                shm = shared_memory.SharedMemory(name=shm_name)
            
            self._shm_blocks[shm_name] = shm
            
            # 3. Copy data
            shm_tensor = torch.frombuffer(shm.buf, dtype=tensor.dtype).reshape(tensor.shape)
            shm_tensor[:] = tensor[:]
            
        # Cache metadata
        self._shm_metadata[shm_name] = (tensor.shape, tensor.dtype)
            
        return SharedMemoryInfo(
            name=shm_name,
            shape=tensor.shape,
            dtype=tensor.dtype,
            image_name=image_name,
            scale=resolution_scale
        )

    def get_image_from_shm(self, info: SharedMemoryInfo) -> torch.Tensor:
        """
        Read image from Shared Memory.
        To be called by reader processes (Rank > 0).
        """
        shm_name = info.name
        
        # Attach to existing SHM if not already attached
        if shm_name not in self._shm_blocks:
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                self._shm_blocks[shm_name] = shm
            except FileNotFoundError:
                raise RuntimeError(f"Shared memory block {shm_name} not found.")
        else:
            shm = self._shm_blocks[shm_name]
            
        # Create tensor view
        tensor = torch.frombuffer(shm.buf, dtype=info.dtype).reshape(info.shape)
        
        # Return the view (zero-copy). 
        # Caller must convert to float if needed.
        return tensor
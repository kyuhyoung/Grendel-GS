#!/usr/bin/env python3
"""
Test script to verify the projection fix in adaptive_tile_utils.py
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class TileBBox:
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @classmethod
    def from_string(cls, bbox_str: str) -> "TileBBox":
        parts = [float(x) for x in bbox_str.split(",")]
        return cls(*parts)

    def get_corners(self) -> np.ndarray:
        return np.array([
            [self.x_min, self.y_min, self.z_min],
            [self.x_min, self.y_min, self.z_max],
            [self.x_min, self.y_max, self.z_min],
            [self.x_min, self.y_max, self.z_max],
            [self.x_max, self.y_min, self.z_min],
            [self.x_max, self.y_min, self.z_max],
            [self.x_max, self.y_max, self.z_min],
            [self.x_max, self.y_max, self.z_max],
        ], dtype=np.float32)

    def to_string(self) -> str:
        return f"{self.x_min},{self.y_min},{self.z_min},{self.x_max},{self.y_max},{self.z_max}"

    @property
    def size_x(self): return self.x_max - self.x_min
    @property
    def size_y(self): return self.y_max - self.y_min
    @property
    def size_z(self): return self.z_max - self.z_min


@dataclass
class CropRegion:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self): return self.x_max - self.x_min
    @property
    def height(self): return self.y_max - self.y_min

    def is_valid(self, img_width: int, img_height: int) -> bool:
        return (self.x_min < img_width and self.x_max > 0 and
                self.y_min < img_height and self.y_max > 0 and
                self.width > 0 and self.height > 0)

    def clamp(self, img_width: int, img_height: int) -> "CropRegion":
        return CropRegion(
            x_min=max(0, self.x_min),
            y_min=max(0, self.y_min),
            x_max=min(img_width, self.x_max),
            y_max=min(img_height, self.y_max),
        )


def project_points_to_camera(points_3d, full_proj_transform, img_width, img_height):
    """NEW FIXED: full_proj_transform만 사용"""
    N = points_3d.shape[0]
    points_h = np.concatenate([points_3d, np.ones((N, 1))], axis=1)
    points_clip = points_h @ full_proj_transform.T

    w = points_clip[:, 3:4]
    valid_mask = (w[:, 0] > 0.001)

    points_ndc = np.zeros((N, 2))
    points_ndc[valid_mask] = points_clip[valid_mask, :2] / w[valid_mask]

    points_2d = np.zeros((N, 2))
    points_2d[:, 0] = (points_ndc[:, 0] + 1.0) * 0.5 * img_width
    points_2d[:, 1] = (points_ndc[:, 1] + 1.0) * 0.5 * img_height

    return points_2d, valid_mask


def old_buggy_projection(points_3d, view_matrix, full_proj, img_width, img_height):
    """OLD BUGGY: view를 두 번 적용"""
    N = points_3d.shape[0]
    points_h = np.concatenate([points_3d, np.ones((N, 1))], axis=1)

    # BUG: view 한 번
    points_cam = points_h @ view_matrix.T
    # BUG: full_proj에 view가 또 들어있음
    points_clip = points_cam @ full_proj.T

    w = points_clip[:, 3:4]
    valid_mask = (w[:, 0] > 0.001)

    points_ndc = np.zeros((N, 2))
    points_ndc[valid_mask] = points_clip[valid_mask, :2] / w[valid_mask]

    points_2d = np.zeros((N, 2))
    points_2d[:, 0] = (points_ndc[:, 0] + 1.0) * 0.5 * img_width
    points_2d[:, 1] = (points_ndc[:, 1] + 1.0) * 0.5 * img_height

    return points_2d, valid_mask


def compute_crop(corners_2d, valid_mask, img_width, img_height, margin=100):
    if not np.any(valid_mask):
        return None
    valid_corners = corners_2d[valid_mask]
    x_min = int(np.floor(valid_corners[:, 0].min())) - margin
    y_min = int(np.floor(valid_corners[:, 1].min())) - margin
    x_max = int(np.ceil(valid_corners[:, 0].max())) + margin
    y_max = int(np.ceil(valid_corners[:, 1].max())) + margin
    crop = CropRegion(x_min, y_min, x_max, y_max)
    if not crop.is_valid(img_width, img_height):
        return None
    return crop.clamp(img_width, img_height)


def create_camera_matrices(img_width, img_height, fov_deg, cam_pos, look_at=(0,0,0)):
    import math
    cam_pos = np.array(cam_pos)
    look_at = np.array(look_at)

    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 0.001:
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    R = np.array([right, up, -forward])
    T = -R @ cam_pos

    world_view = np.eye(4)
    world_view[:3, :3] = R
    world_view[:3, 3] = T

    fov_rad = math.radians(fov_deg)
    tanfov = math.tan(fov_rad / 2)
    aspect = img_width / img_height
    near, far = 0.01, 100.0

    proj = np.zeros((4, 4))
    proj[0, 0] = 1.0 / (aspect * tanfov)
    proj[1, 1] = 1.0 / tanfov
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0

    full_proj = world_view @ proj
    return world_view, full_proj


def main():
    print("=" * 70)
    print("프로젝션 버그 수정 테스트")
    print("=" * 70)

    # ========================================
    # 실제 로그에서 가져온 값들
    # ========================================

    # 전체 씬의 포인트 클라우드 범위 (compute_initial_bbox에서 계산됨)
    # 이것이 초기 타일 영역으로 사용됨
    scene_bbox = TileBBox.from_string(
        "-0.70278000831604,-1.6282552003860473,-1.5714396238327026,"
        "0.8235735893249512,1.0576310276985168,0.5176650285720825"
    )

    # 현재 타일 = 전체 씬 (아직 분할 전)
    tile_bbox = scene_bbox

    # 이미지 크기 (실제 데이터셋)
    img_width = 5472
    img_height = 3648

    print(f"\n[1] 전체 씬 (포인트 클라우드) 영역:")
    print(f"    X: {scene_bbox.x_min:.3f} ~ {scene_bbox.x_max:.3f} (크기: {scene_bbox.size_x:.3f})")
    print(f"    Y: {scene_bbox.y_min:.3f} ~ {scene_bbox.y_max:.3f} (크기: {scene_bbox.size_y:.3f})")
    print(f"    Z: {scene_bbox.z_min:.3f} ~ {scene_bbox.z_max:.3f} (크기: {scene_bbox.size_z:.3f})")

    print(f"\n[2] 현재 타일 영역:")
    print(f"    X: {tile_bbox.x_min:.3f} ~ {tile_bbox.x_max:.3f} (크기: {tile_bbox.size_x:.3f})")
    print(f"    Y: {tile_bbox.y_min:.3f} ~ {tile_bbox.y_max:.3f} (크기: {tile_bbox.size_y:.3f})")
    print(f"    Z: {tile_bbox.z_min:.3f} ~ {tile_bbox.z_max:.3f} (크기: {tile_bbox.size_z:.3f})")
    print(f"    → 타일이 전체 씬의 100%를 커버함 (아직 분할 안 됨)")

    print(f"\n[3] 이미지 크기: {img_width} x {img_height}")

    # ========================================
    # 카메라 시뮬레이션
    # ========================================
    print(f"\n[4] 카메라 시뮬레이션 (FoV=60도, 원점 바라봄):")

    cam_positions = [
        (0, 0, 5),      # 정면
        (5, 0, 0),      # 측면
        (3, 2, 3),      # 대각선
    ]

    for cam_pos in cam_positions:
        world_view, full_proj = create_camera_matrices(
            img_width, img_height, fov_deg=60.0, cam_pos=cam_pos
        )

        corners = tile_bbox.get_corners()

        # OLD (버그)
        old_2d, old_valid = old_buggy_projection(
            corners, world_view, full_proj, img_width, img_height
        )
        old_crop = compute_crop(old_2d, old_valid, img_width, img_height, margin=100)

        # NEW (수정됨)
        new_2d, new_valid = project_points_to_camera(
            corners, full_proj, img_width, img_height
        )
        new_crop = compute_crop(new_2d, new_valid, img_width, img_height, margin=100)

        print(f"\n    카메라 위치: {cam_pos}")

        if old_crop:
            old_pct_w = old_crop.width / img_width * 100
            old_pct_h = old_crop.height / img_height * 100
            print(f"      [OLD 버그] crop: {old_crop.width} x {old_crop.height} ({old_pct_w:.1f}% x {old_pct_h:.1f}%)")
        else:
            print(f"      [OLD 버그] crop: None (안 보임)")

        if new_crop:
            new_pct_w = new_crop.width / img_width * 100
            new_pct_h = new_crop.height / img_height * 100
            print(f"      [NEW 수정] crop: {new_crop.width} x {new_crop.height} ({new_pct_w:.1f}% x {new_pct_h:.1f}%)")
        else:
            print(f"      [NEW 수정] crop: None (안 보임)")

    # ========================================
    # 문제 재현: 로그에서 본 202x203 crop
    # ========================================
    print(f"\n" + "=" * 70)
    print("문제 재현: 왜 crop이 202x203이었나?")
    print("=" * 70)

    print(f"\n전체 씬을 커버하는 타일인데, crop이 202x203 = 아주 작음")
    print(f"margin 100을 빼면 실제 투영 영역은 약 2x3 픽셀!")
    print(f"\n원인: view transform이 두 번 적용됨")
    print(f"  1) world_view_transform으로 한 번")
    print(f"  2) full_proj_transform에 또 포함되어 있어서 또 한 번")
    print(f"\n결과: 3D 포인트가 카메라에서 훨씬 멀리 밀려나서")
    print(f"      투영했을 때 아주 작은 영역만 차지함")

    print(f"\n" + "=" * 70)
    print("수정 후 예상 결과")
    print("=" * 70)
    print(f"\n타일이 전체 씬(100%)을 커버하므로:")
    print(f"  - 대부분의 카메라에서 이미지의 상당 부분을 차지해야 함")
    print(f"  - crop이 수천 픽셀 단위여야 정상")
    print(f"  - 202x203 같은 작은 값은 버그")


if __name__ == "__main__":
    main()

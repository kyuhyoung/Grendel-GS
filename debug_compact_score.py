import numpy as np
import sys
sys.path.append('/workspace/Grendel-GS')

# Load camera positions from COLMAP
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text
from utils.camera_utils import cameraList_from_camInfos

data_path = "/data/samsung_dong"
cameras_extrinsic_file = f"{data_path}/sparse/0/images.txt"
cameras_intrinsic_file = f"{data_path}/sparse/0/cameras.txt"

cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

# Get positions (camera centers in world coordinates)
positions = {}
for idx, key in enumerate(cam_extrinsics):
    extr = cam_extrinsics[key]
    R = np.transpose(extr.qvec2rotmat())
    T = np.array(extr.tvec)
    # Camera center in world coordinates: -R^T * T
    cam_center = -R.T @ T
    cam_id = extr.id
    positions[cam_id] = cam_center

# Get positions for cameras 54, 38, 21, 35, 67
cam_54 = positions[54]
cam_38 = positions[38]
cam_21 = positions[21]
cam_35 = positions[35]
cam_67 = positions[67]

print("=" * 80)
print("Camera Positions:")
print("=" * 80)
print(f"Camera 54: {cam_54}")
print(f"Camera 38: {cam_38}")
print(f"Camera 21: {cam_21}")
print(f"Camera 35: {cam_35}")
print(f"Camera 67: {cam_67}")
print()

# Current window D = [54, 38]
D_positions = np.array([cam_54, cam_38])
D_center = np.mean(D_positions, axis=0)
D_distances = np.linalg.norm(D_positions - D_center, axis=1)  # ← axis=1 (각 카메라별 거리)
D_mean_radius = np.mean(D_distances) + 1e-6

print("=" * 80)
print("Current Window D = [54, 38]:")
print("=" * 80)
print(f"D_center: {D_center}")
print(f"D_distances: {D_distances}")
print(f"D_mean_radius: {D_mean_radius:.6f}")
print()

# Test Camera 21
print("=" * 80)
print("Testing Camera 21:")
print("=" * 80)
D_prime_21 = np.array([cam_54, cam_38, cam_21])
D_prime_center_21 = np.mean(D_prime_21, axis=0)
distances_21 = np.linalg.norm(D_prime_21 - D_prime_center_21, axis=1)
max_distance_21 = np.max(distances_21) + 1e-6
normalized_max_21 = max_distance_21 / D_mean_radius
compact_score_21 = np.exp(-normalized_max_21)

print(f"D' center: {D_prime_center_21}")
print(f"Distances from D' center: {distances_21}")
print(f"Max distance: {max_distance_21:.6f}")
print(f"Normalized max distance: {normalized_max_21:.6f}")
print(f"Compact score: {compact_score_21:.6f} (expected: 0.355)")
print()

# Test Camera 35
print("=" * 80)
print("Testing Camera 35:")
print("=" * 80)
D_prime_35 = np.array([cam_54, cam_38, cam_35])
D_prime_center_35 = np.mean(D_prime_35, axis=0)
distances_35 = np.linalg.norm(D_prime_35 - D_prime_center_35, axis=1)
max_distance_35 = np.max(distances_35) + 1e-6
normalized_max_35 = max_distance_35 / D_mean_radius
compact_score_35 = np.exp(-normalized_max_35)

print(f"D' center: {D_prime_center_35}")
print(f"Distances from D' center: {distances_35}")
print(f"Max distance: {max_distance_35:.6f}")
print(f"Normalized max distance: {normalized_max_35:.6f}")
print(f"Compact score: {compact_score_35:.6f} (expected: 0.131)")
print()

# Test Camera 67
print("=" * 80)
print("Testing Camera 67:")
print("=" * 80)
D_prime_67 = np.array([cam_54, cam_38, cam_67])
D_prime_center_67 = np.mean(D_prime_67, axis=0)
distances_67 = np.linalg.norm(D_prime_67 - D_prime_center_67, axis=1)
max_distance_67 = np.max(distances_67) + 1e-6
normalized_max_67 = max_distance_67 / D_mean_radius
compact_score_67 = np.exp(-normalized_max_67)

print(f"D' center: {D_prime_center_67}")
print(f"Distances from D' center: {distances_67}")
print(f"Max distance: {max_distance_67:.6f}")
print(f"Normalized max distance: {normalized_max_67:.6f}")
print(f"Compact score: {compact_score_67:.6f} (log: 0.033)")
print()

# Summary
print("=" * 80)
print("SUMMARY:")
print("=" * 80)
print(f"Camera 21 compact score: {compact_score_21:.6f} (log: 0.129)")
print(f"Camera 35 compact score: {compact_score_35:.6f} (log: 0.130)")
print(f"Camera 67 compact score: {compact_score_67:.6f} (log: 0.033)")
print()
print(f"Why Camera 67 has much lower compact score:")
print(f"  - Camera 21 max_distance from D' center: {max_distance_21:.2f}m")
print(f"  - Camera 35 max_distance from D' center: {max_distance_35:.2f}m")
print(f"  - Camera 67 max_distance from D' center: {max_distance_67:.2f}m")
print(f"  - Camera 67 spreads D' {max_distance_67 / max_distance_21:.2f}x more than Camera 21")

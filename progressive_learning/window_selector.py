"""
Progressive Window Selector for Drone Imagery
Implements spatial coherence-based sliding window selection algorithm
"""

import numpy as np
from typing import List, Set, Tuple, Optional
from collections import deque
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.ops import unary_union


class WindowSelector:
    """
    Selects camera windows for progressive training based on spatial coherence
    """

    def __init__(self, camera_positions: dict,
                 camera_footprints: Optional[dict] = None):
        """
        Args:
            camera_positions: Dict mapping camera_id -> [x, y] position (numpy array)
            camera_footprints: Dict mapping camera_id -> footprint rectangle [4, 2] array
        """
        self.camera_positions = camera_positions  # {cam_id: [x, y]}
        self.camera_footprints = camera_footprints or {}  # footprint rectangles

        # A: Set of remaining camera IDs to process
        self.A = set(camera_positions.keys())

        # D: Current window (camera_id, timestamp) tuples - no max length since it's dynamic
        self.D = deque()

        # Timestamp counter
        self.iteration = 0

        # Global mean F (computed once)
        self.F = None

    def distance(self, cam_id1: int, cam_id2: int) -> float:
        """
        Compute Euclidean distance between two cameras

        Args:
            cam_id1, cam_id2: Camera IDs

        Returns:
            Euclidean distance
        """
        pos1 = self.camera_positions[cam_id1]
        pos2 = self.camera_positions[cam_id2]
        return np.linalg.norm(pos1 - pos2)

    def compute_mean(self, cam_ids: Set[int]) -> np.ndarray:
        """
        Compute mean position of given camera IDs

        Args:
            cam_ids: Set of camera IDs

        Returns:
            [2] array of mean (x, y) position
        """
        if not cam_ids:
            return np.array([0.0, 0.0])
        positions = [self.camera_positions[cam_id] for cam_id in cam_ids]
        return np.mean(positions, axis=0)

    def compute_median(self, cam_ids: Set[int]) -> np.ndarray:
        """
        Compute median position of given camera IDs

        Args:
            cam_ids: Set of camera IDs

        Returns:
            [2] array of median (x, y) position
        """
        if not cam_ids:
            return np.array([0.0, 0.0])
        positions = [self.camera_positions[cam_id] for cam_id in cam_ids]
        return np.median(positions, axis=0)

    def get_nearest_neighbors(self, center_cam_id: int, k: int) -> List[int]:
        """
        Get k nearest neighbors to a given camera (including itself)

        Args:
            center_cam_id: Center camera ID
            k: Number of neighbors to return

        Returns:
            List of k camera IDs (sorted by distance)
        """
        distances = []
        for cam_id in self.A:
            dist = self.distance(center_cam_id, cam_id)
            distances.append((cam_id, dist))

        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return [cam_id for cam_id, _ in distances[:k]]

    def compute_std(self, cam_ids: List[int]) -> float:
        """
        Compute standard deviation of positions for given camera IDs

        Args:
            cam_ids: List of camera IDs

        Returns:
            Standard deviation of positions
        """
        if len(cam_ids) <= 1:
            return 0.0

        positions = np.array([self.camera_positions[cam_id] for cam_id in cam_ids])
        # Compute std of both x and y, then take mean
        std_x = np.std(positions[:, 0])
        std_y = np.std(positions[:, 1])
        return (std_x + std_y) / 2.0

    def compute_convex_hull(self, cam_ids: Set[int]) -> Tuple[Optional[ConvexHull], List[int]]:
        """
        Compute convex hull for given camera IDs

        Args:
            cam_ids: Set of camera IDs

        Returns:
            Tuple of (ConvexHull object or None, list of vertex camera IDs)
        """
        if len(cam_ids) < 3:
            # Not enough points for convex hull, return all points
            return None, list(cam_ids)

        # Get positions
        cam_ids_list = list(cam_ids)
        positions = np.array([self.camera_positions[cam_id] for cam_id in cam_ids_list])

        try:
            # Compute convex hull
            hull = ConvexHull(positions)

            # Get vertex camera IDs
            vertex_cam_ids = [cam_ids_list[i] for i in hull.vertices]

            return hull, vertex_cam_ids
        except Exception as e:
            # Points are collinear or degenerate, return all points
            print(f"  Warning: Convex hull failed ({type(e).__name__}), using all points")
            return None, cam_ids_list

    def select_initial_window(self, n_cameras: int) -> List[int]:
        """
        Select initial N cameras based on spatial density (Algorithm Step 1-5)

        Args:
            n_cameras: Number of cameras to select

        Returns:
            List of N camera IDs forming the most compact cluster
        """

        # Step 1: Compute global mean F
        self.F = self.compute_mean(self.A)
        print(f"\n[Step 1] Global mean F: {self.F}")

        # Step 2: Compute convex hull of A
        hull, C = self.compute_convex_hull(self.A)
        print(f"[Step 2] Convex hull vertices C: {C} ({len(C)} points)")

        # Step 3-4: Find most compact N-point cluster among convex hull vertices
        K = float('inf')  # K = sufficiently large number
        D = []

        print(f"[Step 3-4] Finding most compact cluster of {n_cameras} cameras...")

        for H in C:
            # Step 4.1: Get N nearest neighbors to H (including H)
            J = self.get_nearest_neighbors(H, n_cameras)

            # Step 4.2: Compute std of J
            L = self.compute_std(J)

            print(f"  Candidate H={H}: neighbors={J[:3]}..., std={L:.3f}")

            # Step 4.3: Update if more compact
            if L < K:
                K = L
                D = J
                print(f"    → New best! K={K:.3f}")

        print(f"[Step 5] Selected initial window D: {D}")
        print(f"         Compactness (std): {K:.3f}")

        # Step 5: Remove D from A
        self.A -= set(D)

        return D

    def compute_window_mean(self, cam_ids: List[int]) -> np.ndarray:
        """
        Compute mean position of cameras in window (Step 6: compute P)

        Args:
            cam_ids: List of camera IDs

        Returns:
            [2] array of mean (x, y) position
        """
        if not cam_ids:
            return np.array([0.0, 0.0])

        positions = np.array([self.camera_positions[cam_id] for cam_id in cam_ids])
        P = np.mean(positions, axis=0)

        print(f"[Step 6] Window mean P: ({P[0]:.3f}, {P[1]:.3f})")
        return P

    def compute_window_footprint_union(self, cam_ids: List[int]) -> Optional[Polygon]:
        """
        Compute footprint union Z for cameras in window (Step 7)

        Args:
            cam_ids: List of camera IDs

        Returns:
            Shapely Polygon representing union Z, or None
        """
        print(f"[Step 7] Computing footprint union Z for {len(cam_ids)} cameras...")

        Z = self.compute_footprint_union(cam_ids)

        if Z is not None:
            if Z.geom_type == 'Polygon':
                area = Z.area
                print(f"         Union Z area: {area:.2f} m²")
            elif Z.geom_type == 'MultiPolygon':
                area = sum(poly.area for poly in Z.geoms)
                print(f"         Union Z area: {area:.2f} m² ({len(Z.geoms)} polygons)")
        else:
            print(f"         Warning: Could not compute union Z")

        return Z

    def compute_footprint_union(self, cam_ids: List[int]) -> Optional[Polygon]:
        """
        Compute union of footprint rectangles for given camera IDs

        Args:
            cam_ids: List of camera IDs

        Returns:
            Shapely Polygon representing the union, or None if no footprints
        """
        polygons = []

        for cam_id in cam_ids:
            if cam_id in self.camera_footprints:
                footprint = self.camera_footprints[cam_id]

                # footprint should be [4, 2] array of corner points
                if isinstance(footprint, np.ndarray) and footprint.shape[0] >= 3:
                    try:
                        poly = Polygon(footprint)
                        if poly.is_valid:
                            polygons.append(poly)
                    except Exception as e:
                        print(f"  Warning: Could not create polygon for camera {cam_id}: {e}")

        if not polygons:
            print("  Warning: No valid footprints found")
            return None

        # Compute union
        union_poly = unary_union(polygons)

        return union_poly

    def check_intersection(self, polygon: Polygon, cam_id: int) -> bool:
        """
        Check if a camera's footprint intersects with given polygon

        Args:
            polygon: Shapely Polygon (e.g., union of current window footprints)
            cam_id: Camera ID to check

        Returns:
            True if footprints intersect
        """
        if polygon is None:
            return False

        if cam_id not in self.camera_footprints:
            return False

        footprint = self.camera_footprints[cam_id]

        if isinstance(footprint, np.ndarray) and footprint.shape[0] >= 3:
            try:
                cam_poly = Polygon(footprint)
                return cam_poly.intersects(polygon)
            except Exception:
                return False

        return False

    def find_next_camera_E(self, Z: Polygon, F: np.ndarray) -> Optional[int]:
        """
        Find camera E: footprint intersects Z and farthest from global mean F (Step 9)

        Args:
            Z: Union of current window footprints (Shapely Polygon)
            F: Global mean position [2]

        Returns:
            Camera ID of E, or None if no valid candidates
        """
        if Z is None:
            print("[Step 9] Error: Union Z is None")
            return None

        print(f"[Step 9] Finding camera E (intersects Z, farthest from F)...")
        print(f"         Checking {len(self.A)} remaining cameras in A")

        # Find all candidates that intersect with Z
        S = []  # Candidate camera IDs
        for cam_id in self.A:
            if self.check_intersection(Z, cam_id):
                S.append(cam_id)

        if not S:
            print(f"         Error: No cameras intersect with Z")
            return None

        print(f"         Found {len(S)} cameras intersecting with Z")

        # Find camera farthest from F among S
        max_distance = -1.0
        E = None

        for cam_id in S:
            pos = self.camera_positions[cam_id]
            # Calculate Euclidean distance between pos and F
            dist = np.linalg.norm(pos - F)

            if dist > max_distance:
                max_distance = dist
                E = cam_id

        if E is not None:
            pos = self.camera_positions[E]
            print(f"         Selected E = camera ID {E}")
            print(f"         Position: ({pos[0]:.3f}, {pos[1]:.3f})")
            print(f"         Distance from F: {max_distance:.3f}")

        return E

    def find_closest_to_window_mean(self, window_cam_ids: List[int]) -> Optional[int]:
        """
        Find camera E closest to the mean of window cameras (for Window 1)

        Args:
            window_cam_ids: List of camera IDs in initial window

        Returns:
            Camera ID of E (closest to window mean), or None if not found
        """
        if not window_cam_ids:
            print("[find_closest_to_window_mean] Error: window_cam_ids is empty")
            return None

        # Compute mean position of window cameras
        window_positions = [self.camera_positions[cam_id] for cam_id in window_cam_ids]
        window_mean = np.mean(window_positions, axis=0)

        print(f"[find_closest_to_window_mean] Finding camera closest to window mean...")
        print(f"         Window cameras: {window_cam_ids}")
        print(f"         Window mean: ({window_mean[0]:.3f}, {window_mean[1]:.3f})")
        print(f"         Checking {len(self.A)} remaining cameras in A")

        # Find camera in A closest to window mean
        min_distance = float('inf')
        E = None

        for cam_id in self.A:
            pos = self.camera_positions[cam_id]
            dist = np.linalg.norm(pos - window_mean)

            if dist < min_distance:
                min_distance = dist
                E = cam_id

        if E is not None:
            pos = self.camera_positions[E]
            print(f"         Selected E = camera ID {E}")
            print(f"         Position: ({pos[0]:.3f}, {pos[1]:.3f})")
            print(f"         Distance from window mean: {min_distance:.3f}")
        else:
            print(f"         Error: No camera found in A")

        return E

    def find_farthest_from_camera(self, cam_id: int, D_cam_ids: List[int]) -> Optional[int]:
        """
        Find camera G in D that is farthest from given camera.

        Args:
            cam_id: Reference camera ID (e.g., E)
            D_cam_ids: List of camera IDs in current window D

        Returns:
            Camera ID of G (farthest from cam_id), or None if not found
        """
        print(f"\n   🔍 Finding farthest camera from camera {cam_id} in D")
        print(f"      D has {len(D_cam_ids)} cameras: {D_cam_ids}")

        if len(D_cam_ids) == 0:
            print(f"      ❌ D is empty, cannot find G")
            return None

        ref_pos = self.camera_positions[cam_id]
        max_distance = -1
        G = None

        for d_cam_id in D_cam_ids:
            pos = self.camera_positions[d_cam_id]
            dist = np.linalg.norm(pos - ref_pos)

            if dist > max_distance:
                max_distance = dist
                G = d_cam_id

        if G is not None:
            pos = self.camera_positions[G]
            print(f"         Selected G = camera ID {G}")
            print(f"         Position: ({pos[0]:.3f}, {pos[1]:.3f})")
            print(f"         Distance from camera {cam_id}: {max_distance:.3f}")

        return G


def test_basic_functions():
    """Test basic utility functions"""
    print("=" * 60)
    print("Testing Basic Utility Functions")
    print("=" * 60)

    # Create sample camera positions (10 cameras in a line)
    positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
        [6.0, 0.0],
        [7.0, 0.0],
        [8.0, 0.0],
        [9.0, 0.0],
    ])
    camera_ids = list(range(10))
    window_size = 5

    selector = WindowSelector(positions, camera_ids, window_size)

    # Test 1: Distance
    print("\n[Test 1] Distance computation:")
    dist = selector.distance(0, 3)
    print(f"  Distance between cam 0 and cam 3: {dist:.2f} (expected: 3.00)")
    assert abs(dist - 3.0) < 0.01, "Distance test failed!"

    # Test 2: Mean
    print("\n[Test 2] Mean computation:")
    mean = selector.compute_mean({0, 1, 2, 3, 4})
    print(f"  Mean of cams 0-4: {mean} (expected: [2.0, 0.0])")
    assert np.allclose(mean, [2.0, 0.0]), "Mean test failed!"

    # Test 3: Median
    print("\n[Test 3] Median computation:")
    median = selector.compute_median({0, 1, 2, 3, 4})
    print(f"  Median of cams 0-4: {median} (expected: [2.0, 0.0])")
    assert np.allclose(median, [2.0, 0.0]), "Median test failed!"

    # Test 4: Nearest neighbors
    print("\n[Test 4] Nearest neighbors:")
    neighbors = selector.get_nearest_neighbors(5, 3)
    print(f"  3 nearest neighbors to cam 5: {neighbors} (expected: [5, 4, 6] or [5, 6, 4])")
    assert 5 in neighbors and (4 in neighbors or 6 in neighbors), "Nearest neighbors test failed!"

    # Test 5: Standard deviation
    print("\n[Test 5] Standard deviation:")
    std = selector.compute_std([0, 1, 2, 3, 4])
    print(f"  Std of cams 0-4: {std:.2f} (expected: ~0.71)")
    assert abs(std - 0.71) < 0.1, "Std test failed!"

    print("\n" + "=" * 60)
    print("✅ All basic function tests passed!")
    print("=" * 60)


def test_convex_hull():
    """Test convex hull computation"""
    print("\n" + "=" * 60)
    print("Testing Convex Hull Computation")
    print("=" * 60)

    # Create sample camera positions forming a square with some interior points
    positions = np.array([
        [0.0, 0.0],   # 0: corner
        [4.0, 0.0],   # 1: corner
        [4.0, 4.0],   # 2: corner
        [0.0, 4.0],   # 3: corner
        [2.0, 2.0],   # 4: interior
        [1.0, 1.0],   # 5: interior
        [3.0, 3.0],   # 6: interior
    ])
    camera_ids = list(range(7))
    window_size = 5

    selector = WindowSelector(positions, camera_ids, window_size)

    # Test 1: Convex hull of all points
    print("\n[Test 1] Convex hull of all 7 points:")
    hull, vertices = selector.compute_convex_hull(set(range(7)))
    print(f"  Hull vertices: {sorted(vertices)} (expected: [0, 1, 2, 3])")
    assert set(vertices) == {0, 1, 2, 3}, "Convex hull test failed!"

    # Test 2: Convex hull of interior points only (may be collinear)
    print("\n[Test 2] Convex hull of interior points only (may be collinear):")
    hull, vertices = selector.compute_convex_hull({4, 5, 6})
    print(f"  Hull vertices: {sorted(vertices)} (expected: all 3 points if collinear)")
    assert len(vertices) == 3, "Interior points hull test failed!"

    # Test 3: Convex hull of 2 points (degenerate case)
    print("\n[Test 3] Convex hull of 2 points (degenerate):")
    hull, vertices = selector.compute_convex_hull({0, 1})
    print(f"  Hull: {hull}, vertices: {sorted(vertices)} (expected: None, [0, 1])")
    assert hull is None and set(vertices) == {0, 1}, "Degenerate hull test failed!"

    print("\n" + "=" * 60)
    print("✅ All convex hull tests passed!")
    print("=" * 60)


def test_initial_window_selection():
    """Test initial window selection based on density"""
    print("\n" + "=" * 60)
    print("Testing Initial Window Selection (Step 1-5)")
    print("=" * 60)

    # Create sample camera positions: dense cluster on left, sparse on right
    positions = np.array([
        # Dense cluster (0-4)
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.5, 0.5],
        [1.0, 0.5],
        # Sparse points (5-9)
        [5.0, 0.0],
        [10.0, 0.0],
        [15.0, 0.0],
        [20.0, 0.0],
        [25.0, 0.0],
    ])
    camera_ids = list(range(10))
    window_size = 5

    selector = WindowSelector(positions, camera_ids, window_size)

    # Test: Select initial window
    print("\n[Test] Selecting initial window of 5 cameras:")
    initial_window = selector.select_initial_window()

    print(f"\n  Result: {initial_window}")
    print(f"  Remaining A: {sorted(selector.A)}")

    # The dense cluster (0-4) should be selected
    assert len(initial_window) == window_size, "Window size mismatch!"
    assert len(selector.A) == 5, "Remaining cameras count mismatch!"

    # Check that selected cameras are from dense cluster
    dense_cluster = {0, 1, 2, 3, 4}
    selected_set = set(initial_window)
    overlap = len(selected_set & dense_cluster)
    print(f"  Overlap with dense cluster {dense_cluster}: {overlap}/5")
    assert overlap >= 4, "Should select mostly from dense cluster!"

    print("\n" + "=" * 60)
    print("✅ Initial window selection test passed!")
    print("=" * 60)


def main():
    """Main function to test Step 1-5 with real-like data"""
    print("=" * 80)
    print("PROGRESSIVE WINDOW SELECTOR - Testing Step 1-5 Only")
    print("=" * 80)

    # Create realistic drone trajectory data (20 cameras)
    np.random.seed(42)

    # Simulate drone flying in a path with some clusters
    positions = []

    # Cluster 1: Dense area (0-6)
    for i in range(7):
        x = i * 0.5 + np.random.normal(0, 0.1)
        y = np.random.normal(0, 0.1)
        positions.append([x, y])

    # Cluster 2: Medium density (7-13)
    for i in range(7):
        x = 5.0 + i * 1.0 + np.random.normal(0, 0.2)
        y = 2.0 + np.random.normal(0, 0.2)
        positions.append([x, y])

    # Cluster 3: Sparse (14-19)
    for i in range(6):
        x = 15.0 + i * 2.0 + np.random.normal(0, 0.3)
        y = 5.0 + np.random.normal(0, 0.3)
        positions.append([x, y])

    positions = np.array(positions)
    camera_ids = list(range(len(positions)))
    window_size = 8

    print(f"\nDataset Info:")
    print(f"  Total cameras: {len(camera_ids)}")
    print(f"  Window size: {window_size}")
    print(f"  Position range: X=[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}], "
          f"Y=[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")

    # Create selector
    selector = WindowSelector(positions, camera_ids, window_size)

    print("\n" + "=" * 80)
    print("RUNNING ALGORITHM STEP 1-5: Select Initial Window")
    print("=" * 80)

    # Run Step 1-5
    initial_window = selector.select_initial_window()

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nInitial Window (D): {initial_window}")
    print(f"  Size: {len(initial_window)}")
    print(f"  Camera IDs: {sorted(initial_window)}")

    # Show positions of selected cameras
    selected_positions = positions[initial_window]
    print(f"\nSelected Camera Positions:")
    for cam_id in initial_window:
        pos = positions[cam_id]
        print(f"  Camera {cam_id}: ({pos[0]:.3f}, {pos[1]:.3f})")

    print(f"\nRemaining Cameras (A): {sorted(selector.A)}")
    print(f"  Count: {len(selector.A)}")

    print(f"\nGlobal Mean (F): ({selector.F[0]:.3f}, {selector.F[1]:.3f})")

    # Calculate statistics
    selected_mean = np.mean(selected_positions, axis=0)
    selected_std = np.std(selected_positions, axis=0).mean()

    print(f"\nSelected Window Statistics:")
    print(f"  Mean position: ({selected_mean[0]:.3f}, {selected_mean[1]:.3f})")
    print(f"  Std deviation: {selected_std:.3f}")

    print("\n" + "=" * 80)
    print("✅ Step 1-5 completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run unit tests
        test_basic_functions()
        test_convex_hull()
        test_initial_window_selection()
    else:
        # Run main demo
        main()

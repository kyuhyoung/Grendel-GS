#!/usr/bin/env python3
"""
Visualize outward_spiral_compact E selection strategy
Generates an animated GIF showing the algorithm in action
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull
import imageio
import os

# Algorithm parameters
OUTWARD_WEIGHT = 0.04            # Weight for moving window center away from F
COMPACT_WEIGHT = 2.5             # Weight for minimizing window variance
SMOOTH_WINDOW_WEIGHT = 2.8       # Weight for smooth window center trajectory
SMOOTH_CAMERA_WEIGHT = 0.7       # Weight for smooth camera addition trajectory

def generate_random_cameras(n_cameras=50, seed=42):
    """Generate grid-based camera positions with perturbation"""
    np.random.seed(seed)

    # Determine grid size (approximately square grid)
    grid_cols = int(np.ceil(np.sqrt(n_cameras)))
    grid_rows = int(np.ceil(n_cameras / grid_cols))

    # Grid spacing and perturbation amount
    spacing = 15.0
    perturbation_scale = 2.5  # Random noise range: ±2.5 units

    positions = []
    count = 0
    for i in range(grid_rows):
        for j in range(grid_cols):
            if count >= n_cameras:
                break

            # Base grid position (centered at origin)
            x_base = (j - (grid_cols - 1) / 2) * spacing
            y_base = (i - (grid_rows - 1) / 2) * spacing

            # Add random perturbation
            x = x_base + np.random.uniform(-perturbation_scale, perturbation_scale)
            y = y_base + np.random.uniform(-perturbation_scale, perturbation_scale)

            positions.append([x, y])
            count += 1

        if count >= n_cameras:
            break

    positions = np.array(positions)
    return positions

def initialize_window_from_convex_hull(positions, n_cameras):
    """
    Initialize window D using convex hull method (same as progressive_trainer.py)

    Algorithm:
    1. Compute convex hull of all positions
    2. For each convex hull vertex H:
       - Find N nearest neighbors (including H)
       - Compute std of distances from mean
    3. Return the most compact cluster (minimum std)
    """
    n_positions = len(positions)

    # Compute convex hull
    try:
        hull = ConvexHull(positions)
        hull_vertices = list(hull.vertices)
        print(f"Convex hull vertices: {hull_vertices} ({len(hull_vertices)} points)")
    except Exception as e:
        print(f"Warning: Convex hull failed ({e}), using all points")
        hull_vertices = list(range(n_positions))

    # Find most compact cluster among hull vertices
    min_std = float('inf')
    best_cluster = None

    for H in hull_vertices:
        # Get N nearest neighbors to H
        distances = np.linalg.norm(positions - positions[H], axis=1)
        nearest_indices = np.argsort(distances)[:n_cameras]

        # Compute std of cluster
        cluster_positions = positions[nearest_indices]
        cluster_mean = np.mean(cluster_positions, axis=0)
        cluster_std = np.std(np.linalg.norm(cluster_positions - cluster_mean, axis=1))

        print(f"  Candidate H={H}: neighbors={nearest_indices[:3]}..., std={cluster_std:.3f}")

        if cluster_std < min_std:
            min_std = cluster_std
            best_cluster = nearest_indices

    print(f"Selected initial window D: {best_cluster} (std={min_std:.3f})")
    return list(best_cluster)

def calculate_balanced_smooth_score(candidate_idx, D_indices, positions, F, prev_window_center,
                                   prev_movement, window_size, outward_weight, compact_weight,
                                   smooth_window_weight, smooth_camera_weight,
                                   last_added_idx, second_last_added_idx):
    """
    Calculate score balancing four forces:
    1. Outward: Window center moves away from F
    2. Compact: Window variance is minimized
    3. Smooth Window: Window center trajectory is smooth (velocity continuity)
    4. Smooth Camera: Added camera trajectory is smooth (directional continuity)
    """

    # Create new window D' with candidate
    D_prime_indices = list(D_indices) + [candidate_idx]

    # Apply sliding window (FIFO if exceeds window_size)
    if len(D_prime_indices) > window_size:
        D_prime_indices = D_prime_indices[-window_size:]

    D_prime_positions = positions[D_prime_indices]
    D_prime_center = np.mean(D_prime_positions, axis=0)

    # 1. Outward score: Distance from F should increase
    dist_to_F = np.linalg.norm(D_prime_center - F)

    if prev_window_center is not None:
        prev_dist_to_F = np.linalg.norm(prev_window_center - F)
        # Reward moving away from F, penalize moving toward F
        outward_delta = dist_to_F - prev_dist_to_F
        outward_score = outward_delta  # Positive = good, negative = bad
    else:
        # First selection, just use distance
        outward_score = dist_to_F / 100.0  # Normalize

    # 2. Compact score: Window radius (max distance from center) should be small
    distances_from_center = np.linalg.norm(D_prime_positions - D_prime_center, axis=1)
    variance = np.var(distances_from_center)  # Still keep for logging
    max_distance = np.max(distances_from_center) + 1e-6
    # Lower max distance = higher score (smaller window radius)
    compact_score = 1.0 / (1.0 + max_distance)  # Normalize to [0, 1]

    # 3. Smooth Window score: Window center movement direction should be continuous
    smooth_window_score = 0.0
    if prev_window_center is not None and prev_movement is not None:
        # Current movement vector
        current_movement = D_prime_center - prev_window_center
        current_norm = np.linalg.norm(current_movement)
        prev_norm = np.linalg.norm(prev_movement)

        if prev_norm > 1e-6 and current_norm > 1e-6:
            # Normalize vectors
            prev_movement_unit = prev_movement / prev_norm
            current_movement_unit = current_movement / current_norm

            # Cosine similarity between movements
            cos_similarity = np.dot(prev_movement_unit, current_movement_unit)
            cos_similarity = np.clip(cos_similarity, -1.0, 1.0)

            # Score: high when directions are similar (smooth trajectory)
            # Range: [-1, 1] → [0, 1], where 1 = same direction, 0 = opposite
            smooth_window_score = (1 + cos_similarity) / 2.0

    # 4. Smooth Camera score: Added camera direction should be continuous
    smooth_camera_score = 0.0
    if last_added_idx is not None and second_last_added_idx is not None:
        # Previous camera movement: second_last → last
        prev_camera_movement = positions[last_added_idx] - positions[second_last_added_idx]
        # Current camera movement: last → candidate
        current_camera_movement = positions[candidate_idx] - positions[last_added_idx]

        prev_cam_norm = np.linalg.norm(prev_camera_movement)
        current_cam_norm = np.linalg.norm(current_camera_movement)

        if prev_cam_norm > 1e-6 and current_cam_norm > 1e-6:
            # Normalize vectors
            prev_cam_unit = prev_camera_movement / prev_cam_norm
            current_cam_unit = current_camera_movement / current_cam_norm

            # Cosine similarity
            cos_similarity_cam = np.dot(prev_cam_unit, current_cam_unit)
            cos_similarity_cam = np.clip(cos_similarity_cam, -1.0, 1.0)

            # Score: high when directions are similar
            smooth_camera_score = (1 + cos_similarity_cam) / 2.0

    # Weighted sum
    total_score = (outward_weight * outward_score +
                   compact_weight * compact_score +
                   smooth_window_weight * smooth_window_score +
                   smooth_camera_weight * smooth_camera_score)

    return total_score, outward_score, compact_score, smooth_window_score, smooth_camera_score, D_prime_center, variance

def select_next_camera(positions, D_indices, available_indices, F, prev_window_center,
                      prev_movement, window_size, outward_weight, compact_weight,
                      smooth_window_weight, smooth_camera_weight,
                      last_added_idx, second_last_added_idx):
    """Select next camera using balanced smooth trajectory strategy with 4 forces"""

    # Score-based selection
    max_score = -float('inf')
    best_idx = None
    best_center = None
    scores = []

    for idx in available_indices:
        score, out_score, comp_score, smooth_win_score, smooth_cam_score, D_prime_center, variance = calculate_balanced_smooth_score(
            idx, D_indices, positions, F, prev_window_center,
            prev_movement, window_size, outward_weight, compact_weight,
            smooth_window_weight, smooth_camera_weight,
            last_added_idx, second_last_added_idx
        )
        scores.append((idx, score, out_score, comp_score, smooth_win_score, smooth_cam_score, variance))

        if score > max_score:
            max_score = score
            best_idx = idx
            best_center = D_prime_center

    # Current window state (for visualization)
    D_positions = positions[D_indices]
    if len(D_indices) == 1:
        D_positions = D_positions.reshape(1, -1)
    D_center = np.mean(D_positions, axis=0)

    return best_idx, best_center, scores, D_center

def plot_frame(positions, D_indices, available_indices, F, best_idx,
               D_center, step, total_steps, selection_history, prev_window_center, axis_limits, window_center_history):
    """Plot a single frame of the visualization"""

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot all cameras (gray)
    ax.scatter(positions[:, 0], positions[:, 1], c='lightgray', s=50, alpha=0.5, label='All cameras')

    # Add ID labels for all cameras
    for i in range(len(positions)):
        ax.text(positions[i, 0], positions[i, 1], str(i),
                fontsize=10, ha='center', va='center',
                color='black', alpha=0.6, zorder=5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

    # Plot window center trajectory
    if len(window_center_history) > 1:
        center_trajectory = np.array(window_center_history)
        ax.plot(center_trajectory[:, 0], center_trajectory[:, 1],
                c='green', linewidth=3, alpha=0.8, linestyle='-',
                marker='o', markersize=6, markerfacecolor='lightgreen',
                markeredgecolor='darkgreen', markeredgewidth=1.5,
                label='Window center trajectory', zorder=8)

    # Plot trajectory (selection history)
    if len(selection_history) > 1:
        trajectory_positions = positions[selection_history]
        ax.plot(trajectory_positions[:, 0], trajectory_positions[:, 1],
                c='red', linewidth=3, alpha=0.7, linestyle='-',
                marker='o', markersize=8, markerfacecolor='orange',
                markeredgecolor='darkred', markeredgewidth=2,
                label='Selection trajectory', zorder=7)

        # Add arrows to show direction
        for i in range(len(trajectory_positions) - 1):
            dx = trajectory_positions[i+1, 0] - trajectory_positions[i, 0]
            dy = trajectory_positions[i+1, 1] - trajectory_positions[i, 1]
            ax.annotate('', xy=(trajectory_positions[i+1, 0], trajectory_positions[i+1, 1]),
                       xytext=(trajectory_positions[i, 0], trajectory_positions[i, 1]),
                       arrowprops=dict(arrowstyle='->', color='darkred', lw=2, alpha=0.6))

        # Add ID labels for all selected cameras in history (except the current best_idx which will be drawn later)
        for idx in selection_history:
            if idx != best_idx:  # Current selection will be drawn separately
                ax.text(positions[idx, 0], positions[idx, 1], str(idx),
                        fontsize=11, ha='center', va='center',
                        color='white', fontweight='bold', zorder=9,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='orange', alpha=0.85, edgecolor='darkred', linewidth=1.5))

    # Plot available cameras (light blue)
    if len(available_indices) > 0:
        avail_pos = positions[list(available_indices)]
        ax.scatter(avail_pos[:, 0], avail_pos[:, 1], c='lightblue', s=80, alpha=0.7, label='Available')

    # Plot current window D (blue)
    D_positions = positions[D_indices]
    ax.scatter(D_positions[:, 0], D_positions[:, 1], c='blue', s=150,
              edgecolors='darkblue', linewidths=2, label='Window D', zorder=10)

    # Add ID labels for Window D cameras (more visible)
    for idx in D_indices:
        ax.text(positions[idx, 0], positions[idx, 1], str(idx),
                fontsize=12, ha='center', va='center',
                color='white', fontweight='bold', zorder=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='darkblue', alpha=0.8, edgecolor='white', linewidth=1.5))

    # Draw convex hull of window D
    if len(D_indices) >= 3:
        try:
            hull = ConvexHull(D_positions)
            # Draw hull edges
            for simplex in hull.simplices:
                ax.plot(D_positions[simplex, 0], D_positions[simplex, 1],
                       'b-', linewidth=2, alpha=0.6)
            # Close the hull
            hull_vertices = D_positions[hull.vertices]
            hull_vertices = np.vstack([hull_vertices, hull_vertices[0]])
            ax.plot(hull_vertices[:, 0], hull_vertices[:, 1],
                   'b-', linewidth=2, alpha=0.6, label='Window D convex hull')
        except Exception as e:
            # Convex hull may fail for collinear points
            pass

    # Plot window center (green cross)
    if D_center is not None:
        ax.scatter([D_center[0]], [D_center[1]], c='green', s=200, marker='x',
                  linewidths=3, label='Window center', zorder=11)

        # Draw circle around window
        D_std = np.std(np.linalg.norm(D_positions - D_center, axis=1))
        circle = Circle(D_center, D_std * 2, fill=False, edgecolor='green',
                       linestyle='--', linewidth=2, alpha=0.5)
        ax.add_patch(circle)

    # Plot selected camera (red star)
    if best_idx is not None:
        ax.scatter([positions[best_idx, 0]], [positions[best_idx, 1]],
                  c='red', s=300, marker='*', edgecolors='darkred',
                  linewidths=2, label='Selected E', zorder=12)

        # Add ID label for selected camera (most visible)
        ax.text(positions[best_idx, 0], positions[best_idx, 1], str(best_idx),
                fontsize=14, ha='center', va='center',
                color='yellow', fontweight='bold', zorder=13,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.9, edgecolor='yellow', linewidth=2))

        # Draw arrow from window center to selected camera
        if D_center is not None:
            ax.arrow(D_center[0], D_center[1],
                    positions[best_idx, 0] - D_center[0],
                    positions[best_idx, 1] - D_center[1],
                    head_width=3, head_length=2, fc='red', ec='darkred',
                    linewidth=2, alpha=0.7, zorder=9)

    # Plot global center F (black cross)
    ax.scatter([F[0]], [F[1]], c='black', s=200, marker='x',
              linewidths=3, label='Global center F', zorder=11)

    # Draw line from F to current window center
    if D_center is not None:
        ax.plot([F[0], D_center[0]], [F[1], D_center[1]],
               'k--', linewidth=2, alpha=0.3, label='F to window center')

    # Draw line from F to previous window center (if available)
    if prev_window_center is not None:
        ax.plot([F[0], prev_window_center[0]], [F[1], prev_window_center[1]],
               'purple', linewidth=2, linestyle=':', alpha=0.4, label='Previous direction')

    # Configure plot
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'Balanced Smooth Trajectory Selection - Step {step}/{total_steps}\n' +
                f'Outward={OUTWARD_WEIGHT}, Compact={COMPACT_WEIGHT}, SmoothWin={SMOOTH_WINDOW_WEIGHT}, SmoothCam={SMOOTH_CAMERA_WEIGHT}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # Set fixed axis limits for all frames
    if axis_limits:
        ax.set_xlim(axis_limits['x'])
        ax.set_ylim(axis_limits['y'])

    # Add info text
    info_text = f'Window size: {len(D_indices)}\n'
    info_text += f'Available: {len(available_indices)}\n'
    if prev_window_center is not None:
        dist = np.linalg.norm(prev_window_center - F)
        info_text += f'Prev dist from F: {dist:.1f}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save to temporary file
    temp_file = f'/tmp/frame_{step:03d}.png'
    plt.savefig(temp_file, dpi=100)
    plt.close()

    return temp_file

def run_visualization(n_cameras=50, initial_window_size=None, max_steps=None,
                     output_file='spiral_selection.gif', duration=1.0):
    """Run the complete visualization and generate GIF"""

    # If initial_window_size not specified, use 1/10 of cameras
    if initial_window_size is None:
        initial_window_size = max(2, n_cameras // 10)  # At least 2

    # If max_steps not specified, select all cameras
    if max_steps is None:
        max_steps = n_cameras - initial_window_size

    print("="*60)
    print("Balanced Smooth Trajectory Selection Visualization")
    print("="*60)
    print(f"Parameters: Outward={OUTWARD_WEIGHT}, Compact={COMPACT_WEIGHT}, "
          f"SmoothWindow={SMOOTH_WINDOW_WEIGHT}, SmoothCamera={SMOOTH_CAMERA_WEIGHT}")
    print(f"Cameras: {n_cameras}, Initial window: {initial_window_size}, Steps: {max_steps}")
    print()

    # Generate random cameras
    positions = generate_random_cameras(n_cameras)

    # Calculate fixed axis limits for all frames
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    axis_limits = {
        'x': (x_min - x_margin, x_max + x_margin),
        'y': (y_min - y_margin, y_max + y_margin)
    }

    # Calculate global center F
    F = np.mean(positions, axis=0)
    print(f"Global center F: ({F[0]:.2f}, {F[1]:.2f})")

    # Initialize window D using convex hull method (same as progressive_trainer)
    print(f"\nInitializing window D (size={initial_window_size})...")
    initial_indices = initialize_window_from_convex_hull(positions, initial_window_size)

    D_indices = list(initial_indices)
    available_indices = set(range(n_cameras)) - set(D_indices)

    # Initial window center and movement tracking
    D_positions = positions[D_indices]
    prev_window_center = np.mean(D_positions, axis=0)
    prev_movement = None  # No previous movement for first step
    selection_history = list(initial_indices)  # Track all selected cameras in order
    window_center_history = [prev_window_center.copy()]  # Track window center trajectory

    frames = []

    # Generate frames
    for step in range(max_steps):
        print(f"\nStep {step + 1}/{max_steps}")
        print(f"  Window D: {D_indices}")
        print(f"  Available: {len(available_indices)} cameras")

        if len(available_indices) == 0:
            print("  No more cameras available!")
            break

        # Get last two added cameras for smooth camera score
        last_added_idx = selection_history[-1] if len(selection_history) >= 1 else None
        second_last_added_idx = selection_history[-2] if len(selection_history) >= 2 else None

        # Select next camera using balanced smooth trajectory strategy
        best_idx, best_center, scores, D_center = select_next_camera(
            positions, D_indices, available_indices, F, prev_window_center,
            prev_movement, initial_window_size, OUTWARD_WEIGHT, COMPACT_WEIGHT,
            SMOOTH_WINDOW_WEIGHT, SMOOTH_CAMERA_WEIGHT,
            last_added_idx, second_last_added_idx
        )

        if best_idx is None:
            print("  Could not select next camera!")
            break

        # Show top 5 scores for better analysis
        print(f"  Top 5 candidates:")
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for rank, (idx, score, out_score, comp_score, smooth_win_score, smooth_cam_score, variance) in enumerate(sorted_scores[:5], 1):
            marker = "✓ SELECTED" if idx == best_idx else ""
            print(f"    {rank}. Camera {idx}: total={score:.4f} | "
                  f"outward={out_score:.3f} | compact={comp_score:.3f} | "
                  f"smooth_win={smooth_win_score:.3f} | smooth_cam={smooth_cam_score:.3f} | "
                  f"variance={variance:.2f} {marker}")

        print(f"  --- Selection Decision ---")
        # Find selected camera's scores
        selected_score_info = next((s for s in scores if s[0] == best_idx), None)
        if selected_score_info:
            idx, score, out_score, comp_score, smooth_win_score, smooth_cam_score, variance = selected_score_info
            print(f"  ✓ Selected Camera {best_idx}")
            print(f"    Reason: Highest total score = {score:.4f}")
            print(f"    Breakdown:")
            print(f"      - Outward      (×{OUTWARD_WEIGHT}): {out_score:.3f}")
            print(f"      - Compact      (×{COMPACT_WEIGHT}): {comp_score:.3f}")
            print(f"      - Smooth Window (×{SMOOTH_WINDOW_WEIGHT}): {smooth_win_score:.3f}")
            print(f"      - Smooth Camera (×{SMOOTH_CAMERA_WEIGHT}): {smooth_cam_score:.3f}")
            if best_center is not None:
                dist = np.linalg.norm(best_center - F)
                print(f"    New window center distance from F: {dist:.2f}")
        else:
            print(f"  ✓ Selected: Camera {best_idx}")

        # Update selection history (before plotting)
        selection_history.append(best_idx)

        # Add new window center to history
        if best_center is not None:
            window_center_history.append(best_center.copy())

        # Plot frame with trajectory
        frame_file = plot_frame(positions, D_indices, available_indices, F,
                               best_idx, D_center, step + 1, max_steps,
                               selection_history, prev_window_center, axis_limits, window_center_history)
        frames.append(frame_file)

        # Update window
        D_indices.append(best_idx)
        available_indices.remove(best_idx)

        # Keep window size at initial_window_size (sliding window)
        if len(D_indices) > initial_window_size:
            removed = D_indices.pop(0)
            print(f"  Removed camera {removed} from window (FIFO)")

        # Calculate movement vector and update tracking
        if best_center is not None:
            if prev_window_center is not None:
                prev_movement = best_center - prev_window_center
            prev_window_center = best_center

    # Create GIF
    print(f"\n{'='*60}")
    print(f"Creating GIF: {output_file}")

    images = [imageio.imread(f) for f in frames]
    # duration is in milliseconds for GIF
    duration_ms = duration * 1000
    imageio.mimsave(output_file, images, duration=duration_ms, loop=0)

    # Clean up temporary files
    for f in frames:
        os.remove(f)

    print(f"✓ GIF created successfully: {output_file}")
    print(f"  Total frames: {len(frames)}")
    print(f"  File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    print("="*60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize outward_spiral_compact selection')
    parser.add_argument('--n_cameras', type=int, default=50, help='Number of cameras')
    parser.add_argument('--initial_window', type=int, default=None, help='Initial window size (default: n_cameras/10)')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum steps to simulate (default: all cameras)')
    parser.add_argument('--output', type=str, default='spiral_selection.gif', help='Output GIF file')
    parser.add_argument('--outward_weight', type=float, default=0.04, help='Weight for outward movement (default: 0.04)')
    parser.add_argument('--compact_weight', type=float, default=2.5, help='Weight for window compactness (default: 2.5)')
    parser.add_argument('--smooth_window_weight', type=float, default=2.8, help='Weight for smooth window trajectory (default: 2.8)')
    parser.add_argument('--smooth_camera_weight', type=float, default=0.7, help='Weight for smooth camera trajectory (default: 0.7)')
    parser.add_argument('--duration', type=float, default=1.0, help='Duration per frame in seconds (default: 1.0)')

    args = parser.parse_args()

    # Update global parameters (no 'global' needed in module top-level)
    OUTWARD_WEIGHT = args.outward_weight
    COMPACT_WEIGHT = args.compact_weight
    SMOOTH_WINDOW_WEIGHT = args.smooth_window_weight
    SMOOTH_CAMERA_WEIGHT = args.smooth_camera_weight

    run_visualization(
        n_cameras=args.n_cameras,
        initial_window_size=args.initial_window,
        max_steps=args.max_steps,
        output_file=args.output,
        duration=args.duration
    )

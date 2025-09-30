#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import collections
import struct
import cv2
from utils.camera_param_parser import parse_camera_parameters_heuristic

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def generate_tracks_by_projection(points3d, cameras, images):
    """
    Generate track information by projecting 3D points onto camera images.
    Uses batch processing for efficient projection.

    Args:
        points3d: Nx3 array of 3D points
        cameras: Dictionary of camera information
        images: Dictionary of image information

    Returns:
        List of sets, where each set contains camera IDs where the point is visible
    """
    num_points = points3d.shape[0]
    tracks = [set() for _ in range(num_points)]
    '''
    print(f"Projecting {num_points} points to {len(images)} cameras...")
    print(f"🔍 DEBUG: Camera IDs in images: {list(images.keys())}")
    print(f"🔍 DEBUG: Camera IDs in cameras: {list(cameras.keys())}")
    '''
    for image_id, image in images.items():
        camera = cameras[image.camera_id]
        #print(f"🔍 Processing camera {image_id}, camera_id: {image.camera_id}, camera.model : {camera.model}")

        # Support PINHOLE and RADIAL camera models (same as colmap_visualizer.py)
        if camera.model not in ["PINHOLE", "RADIAL"]:
            print(f"⚠️  Skipping unsupported camera model: {camera.model}")
            continue

        # COLMAP coordinate system transformation
        # COLMAP stores qvec (quaternion) and tvec (translation)
        # Convert to rotation matrix using same method as colmap_visualizer.py
        qw, qx, qy, qz = image.qvec  # COLMAP quaternion format: [qw, qx, qy, qz]
        # Use same formula as colmap_visualizer.py
        R_colmap = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
        ])
        T_colmap = np.array(image.tvec)     # COLMAP translation vector

        # COLMAP convention: world_to_camera transformation
        # For cv2.projectPoints, we need world -> camera transformation
        # COLMAP's R and T represent: P_camera = R * (P_world - C) where C is camera center
        # Camera center C = -R^T * T
        # For OpenCV: we use R directly and T = -R * C = T_colmap
        R = R_colmap  # Use COLMAP rotation matrix directly
        T = T_colmap  # Use COLMAP translation vector directly

        # Parse camera parameters using heuristic approach (same as colmap_visualizer.py)
        parsed_params = parse_camera_parameters_heuristic(
            camera.params, camera.width, camera.height, camera.model
        )

        fx = parsed_params['fx']
        fy = parsed_params['fy']
        cx = parsed_params['cx']
        cy = parsed_params['cy']
        distortion = parsed_params['distortion']

        from utils.projection_utils import project_points_to_camera

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Distortion coefficients (pass raw distortion array to util function)
        dist_coeffs = np.array(distortion, dtype=np.float64) if len(distortion) > 0 else None

        # Use common projection utility
        result = project_points_to_camera(
            points3d, R, T, camera_matrix, dist_coeffs,
            check_behind_camera=True,
            image_width=camera.width,
            image_height=camera.height,
            margin_pixels=0
        )

        points_2d = result['points_2d']
        visible_mask = result['visible_mask']

        # Add this image ID to tracks of visible points
        visible_point_indices = np.where(visible_mask)[0]
        #print('aaaaa')
        for point_idx in visible_point_indices:
            tracks[point_idx].add(image_id)  # Use image_id as intended
        #print('bbbbb')
    '''
    # Camera visible points summary (same format as colmap_visualizer.py)
    print("📷 Camera visible points summary:")
    for image_id in sorted(images.keys()):
        count = sum(1 for track in tracks if image_id in track)
        print(f"  Camera {image_id}: {count} visible points")
    print("====================================")
    '''

    return tracks


def read_points3D_text(path, track_by_projection=False, cameras=None, images=None):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    tracks = []
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))

                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error

                # Only parse track info if not using projection-based tracks
                if not track_by_projection:
                    # Track 정보 파싱 (8번째 요소부터 image_id feature_id 쌍들)
                    track_image_ids = set()
                    for i in range(8, len(elems), 2):
                        if i < len(elems):
                            track_image_ids.add(int(elems[i]))
                    tracks.append(track_image_ids)
                else:
                    tracks.append(set())  # Placeholder, will be replaced by projection

                count += 1

    # Use projection-based tracks if requested
    if track_by_projection and cameras is not None and images is not None:
        #print("🎯 Generating tracks by projection...")
        original_tracks = tracks.copy()  # Keep original for comparison
        tracks = generate_tracks_by_projection(xyzs, cameras, images)

        '''
        # Compare results
        original_visible = sum(1 for track in original_tracks if len(track) > 0)
        projection_visible = sum(1 for track in tracks if len(track) > 0)
        print(f"📊 Track comparison: Original COLMAP: {original_visible} visible points, Projection: {projection_visible} visible points")

        # Count points visible per image in projection-based tracks
        image_point_counts = {}
        for image_id in images.keys():
            count = sum(1 for track in tracks if image_id in track)
            image_point_counts[image_id] = count
        print("📷 Projection-based visible points per image:")
        for image_id in sorted(image_point_counts.keys()):
            print(f"  Image {image_id}: {image_point_counts[image_id]} visible points")
        '''
    else:
        # Show COLMAP track-based summary
        print("📷 COLMAP track-based visible points summary:")
        if images is not None:
            for image_id in sorted(images.keys()):
                count = sum(1 for track in tracks if image_id in track)
                print(f"  Camera {image_id}: {count} visible points")
        print("====================================")
    #exit(1)
    return xyzs, rgbs, errors, tracks


def read_points3D_binary(path_to_model_file, track_by_projection=False, cameras=None, images=None):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))
        tracks = []

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )

            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error

            # Only parse track info if not using projection-based tracks
            if not track_by_projection:
                # track_elems는 [image_id1, feature_id1, image_id2, feature_id2, ...] 형태
                track_image_ids = set()
                for i in range(0, len(track_elems), 2):
                    track_image_ids.add(track_elems[i])
                tracks.append(track_image_ids)
            else:
                tracks.append(set())  # Placeholder, will be replaced by projection

    # Use projection-based tracks if requested
    if track_by_projection and cameras is not None and images is not None:
        #print("🎯 Generating tracks by projection...")
        original_tracks = tracks.copy()  # Keep original for comparison
        tracks = generate_tracks_by_projection(xyzs, cameras, images)
        '''
        # Compare results
        original_visible = sum(1 for track in original_tracks if len(track) > 0)
        projection_visible = sum(1 for track in tracks if len(track) > 0)
        print(f"📊 Track comparison: Original COLMAP: {original_visible} visible points, Projection: {projection_visible} visible points")

        # Count points visible per image in projection-based tracks
        image_point_counts = {}
        for image_id in images.keys():
            count = sum(1 for track in tracks if image_id in track)
            image_point_counts[image_id] = count
        print("📷 Projection-based visible points per image:")
        for image_id in sorted(image_point_counts.keys()):
            print(f"  Image {image_id}: {image_point_counts[image_id]} visible points")
        '''

    return xyzs, rgbs, errors, tracks


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                '''
                print(f'path : {path}, model : {model}');  exit(1) # radial
                assert (
                    model == "PINHOLE"
                ), "While the loader support other types, the rest of the code assumes PINHOLE"
                '''
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id, model=model, width=width, height=height, params=params
                )
    return cameras


def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

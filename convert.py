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

import os
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action="store_true")
parser.add_argument("--skip_matching", action="store_true")
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = (
    '"{}"'.format(args.colmap_executable)
    if len(args.colmap_executable) > 0
    else "colmap"
)
magick_command = (
    '"{}"'.format(args.magick_executable)
    if len(args.magick_executable) > 0
    else "magick"
)
use_gpu = 1  # Force GPU usage

# Check if we should skip feature extraction
skip_extraction = os.environ.get('SKIP_EXTRACTION_FLAG', 'false').lower() == 'true'

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Extract GPS coordinates if available
    print("Extracting GPS coordinates from images...")
    gps_file = args.source_path + "/gps_coords.txt"
    extract_gps_cmd = f"python extract_gps.py --image_path {args.source_path}/input --output {gps_file}"
    os.system(extract_gps_cmd)
    
    ## Feature extraction
    if not skip_extraction:
        print(f"=== FEATURE EXTRACTION CONFIGURATION ===")
        print(f"GPU Usage: {'ENABLED' if use_gpu else 'DISABLED'}")
        print(f"GPU Index: 0")
        print(f"Feature Type: SIFT")
        print(f"Camera Model: {args.camera}")
        print(f"============================================")
        
        feat_extracton_cmd = (
            colmap_command + " feature_extractor "
            "--database_path "
            + args.source_path
            + "/distorted/database.db \
            --image_path "
            + args.source_path
            + "/input \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model "
            + args.camera
            + " \
            --FeatureExtraction.type SIFT \
            --FeatureExtraction.use_gpu "
            + str(use_gpu)
            + " \
            --FeatureExtraction.gpu_index 0"
        )
        print(f"Executing command: {feat_extracton_cmd}")
        # Execute without filtering to ensure no data is lost
        exit_code = os.system(feat_extracton_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)
    else:
        print("=== SKIPPING FEATURE EXTRACTION ===")
        print("Proceeding directly to feature matching...")

    ## Feature matching
    print(f"=== FEATURE MATCHING CONFIGURATION ===")
    print(f"GPU Usage: {'ENABLED' if use_gpu else 'DISABLED'}")
    print(f"==========================================")
    
    feat_matching_cmd = (
        colmap_command
        + " exhaustive_matcher \
        --database_path "
        + args.source_path
        + "/distorted/database.db \
        --FeatureMatching.use_gpu "
        + str(use_gpu)
    )
    print(f"Executing matching command: {feat_matching_cmd}")
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (
        colmap_command
        + " mapper \
        --database_path "
        + args.source_path
        + "/distorted/database.db \
        --image_path "
        + args.source_path
        + "/input \
        --output_path "
        + args.source_path
        + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001"
    )
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (
    colmap_command
    + " image_undistorter \
    --image_path "
    + args.source_path
    + "/input \
    --input_path "
    + args.source_path
    + "/distorted/sparse/0 \
    --output_path "
    + args.source_path
    + "\
    --output_type COLMAP"
)
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == "0":
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if args.resize:
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(
            magick_command + " mogrify -resize 50% " + destination_file
        )
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(
            magick_command + " mogrify -resize 25% " + destination_file
        )
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(
            magick_command + " mogrify -resize 12.5% " + destination_file
        )
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")

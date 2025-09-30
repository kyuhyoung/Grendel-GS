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

"""
Camera parameter parsing utilities for COLMAP data.
"""


def parse_camera_parameters_heuristic(raw_params, width, height, model="UNKNOWN"):
    """
    Parse camera parameters using heuristic approach from colmap_visualizer.py.

    Args:
        raw_params: List of raw parameter values from COLMAP
        width: Image width
        height: Image height
        model: Camera model name (for debugging)

    Returns:
        dict: Parsed parameters with 'fx', 'fy', 'cx', 'cy', 'distortion'
    """
    params = {}

    if len(raw_params) >= 4:
        # 4개 이상: fx, fy, cx, cy 순서 가능성 (PINHOLE 등)
        # cx, cy가 width/2, height/2에 가까운지 체크
        potential_cx_cy_pairs = [
            (raw_params[2], raw_params[3]),  # 일반적인 fx,fy,cx,cy 순서
            (raw_params[1], raw_params[2]),  # f,cx,cy,... 순서
        ]

        best_match = None
        best_score = float('inf')

        for i, (cx, cy) in enumerate(potential_cx_cy_pairs):
            cx_error = abs(cx - width/2)
            cy_error = abs(cy - height/2)
            score = cx_error + cy_error

            if score < best_score:
                best_score = score
                best_match = i

        if best_match == 0:  # fx,fy,cx,cy 순서
            params = {
                'fx': raw_params[0],
                'fy': raw_params[1],
                'cx': raw_params[2],
                'cy': raw_params[3],
                'distortion': raw_params[4:] if len(raw_params) > 4 else []
            }
        elif best_match == 1:  # f,cx,cy 순서
            params = {
                'fx': raw_params[0],
                'fy': raw_params[0],  # 단일 focal length
                'cx': raw_params[1],
                'cy': raw_params[2],
                'distortion': raw_params[3:] if len(raw_params) > 3 else []
            }
    else:
        # 3개: f, cx, cy 순서 가능성
        if len(raw_params) == 3:
            params = {
                'fx': raw_params[0],
                'fy': raw_params[0],
                'cx': raw_params[1],
                'cy': raw_params[2],
                'distortion': []
            }

    return params
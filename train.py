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
import sys
print("TRAIN_DEBUG: train.py started - imports beginning", file=sys.stderr, flush=True)
import torch
print("TRAIN_DEBUG: torch imported successfully", file=sys.stderr, flush=True)
import json
from utils.general_utils import safe_state, init_distributed
import utils.general_utils as utils
from argparse import ArgumentParser
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    DistributionParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)
import train_internal

if __name__ == "__main__":
    print("TRAIN_DEBUG: train.py main function started", file=sys.stderr, flush=True)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    print("TRAIN_DEBUG: ArgumentParser created", file=sys.stderr, flush=True)
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    args = parser.parse_args(sys.argv[1:])

    #print(f'\n\n\n args : {args} \n\n\n')
    # Set up distributed training
    init_distributed(args)

    ## Prepare arguments.
    # Check arguments
    init_args(args)

    args = utils.get_args()

    # create log folder
    if utils.GLOBAL_RANK == 0:
        os.makedirs(args.log_folder, exist_ok=True)
        os.makedirs(args.model_path, exist_ok=True)
    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(
            group=utils.DEFAULT_GROUP
        )  # log_folder is created before other ranks start writing log.
    if utils.GLOBAL_RANK == 0:
        with open(args.log_folder + "/args.json", "w") as f:
            json.dump(vars(args), f)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Load previous training state if provided (for progressive training)
    previous_state = None
    is_progressive_training = hasattr(args, 'previous_state') and args.previous_state is not None

    if is_progressive_training:
        if args.previous_state and os.path.exists(args.previous_state):
            # Non-empty path - load from previous window
            try:
                with open(args.previous_state, 'r') as f:
                    previous_state = json.load(f)
                utils.print_rank_0(f"📖 Loaded previous state from: {args.previous_state}")
                utils.print_rank_0(f"   Window: {previous_state.get('iteration_name', 'unknown')}")
                utils.print_rank_0(f"   Processed cameras: {len(previous_state.get('processed_cameras', []))}")
                utils.print_rank_0(f"   Unprocessed points: {len(previous_state.get('unprocessed_points', []))}")
            except Exception as e:
                utils.print_rank_0(f"⚠️  Warning: Could not load previous state: {e}")
                previous_state = None
        else:
            # Empty path - initial window of progressive training
            utils.print_rank_0(f"📖 Progressive training: Initial window (no previous state)")
            previous_state = None

    # Store previous state in args for train_internal access
    # Also store whether this is progressive training (even for initial window)
    args.previous_state_data = previous_state
    args.is_progressive_training = is_progressive_training

    # For compatibility with existing progressive dataset logic
    # Load progressive dataset from temp files if previous state is available
    if previous_state:
        # Progressive training mode - dataset should be loaded from temp files
        # The dataset files are saved by progressive_trainer in temp_* directories
        utils.print_rank_0(f"🔄 Progressive training mode detected - using temporary dataset files")
        args.progressive_dataset = True  # Flag to indicate progressive training mode
    else:
        args.progressive_dataset = None

    # Initialize log file and print all args
    log_file = open(
        args.log_folder
        + "/python_ws="
        + str(utils.WORLD_SIZE)
        + "_rk="
        + str(utils.GLOBAL_RANK)
        + ".log",
        "a" if args.auto_start_checkpoint else "w",
    )
    utils.set_log_file(log_file)
    print_all_args(args, log_file)

    train_internal.training_refactored_main(
        lp.extract(args), op.extract(args), pp.extract(args), args, log_file
    )

    # All done
    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
        # Properly cleanup distributed training
        torch.distributed.destroy_process_group()
    utils.print_rank_0("\nTraining complete.")

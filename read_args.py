import sys
import os
import argparse

import torch

def get_args():
    parser = argparse.ArgumentParser("Foundation Models for Point Tracking - Adaptation")

    # === Data Related Parameters ===
    parser.add_argument('--movi_f_root', type=str, default=None)
    parser.add_argument('--tapvid_root', type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, choices=["davis_first", "davis_strided", "rgb_stacking_first", "rgb_stacking_strided", "kinetics_first", "kinetics_strided"], default="davis_first")
    parser.add_argument('--final_feature_size', type=int, nargs=2, default=[32, 32])
    parser.add_argument('--augmentation', action="store_true")

    # === Model Parameters ===
    parser.add_argument('--mode', type=str, choices=["probe", "lora"])
    parser.add_argument('--backbone', type=str, choices=["dinov2_s_14",
                                                         "dinov2_b_14", 
                                                         "dinov2_l_14", 
                                                         "dinov2_g_14"], default="dinov2_b_14")
    parser.add_argument('--lora_rank', type=int, default=32)

    # === Training Related Parameters ===
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--T', type=int, default=24)
    parser.add_argument('--epoch_num', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--bs', type=int, default=16)

    parser.add_argument('--model_save_path', type=str, default=None) 
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    args.gpus = torch.cuda.device_count()

    # Extra settings for the models
    args.patch_size = 8 if args.backbone == "sd" else int(args.backbone.split("_")[-1])
    args.input_size = [args.patch_size * fs for fs in args.final_feature_size]

    if args.model_save_path is not None:
        os.makedirs(args.model_save_path, exist_ok=True)

    assert os.path.exists(args.tapvid_root), f"Path {args.tapvid_root} does not exist."
    assert os.path.exists(args.movi_f_root), f"Path {args.movi_f_root} does not exist."

    return args


import sys
import os
import argparse

import numpy as np
from tqdm import tqdm
import time
import datetime
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn


from utils.utils import calculate_cost_volume
from utils.eval_utils import Evaluator, compute_tapvid_metrics

from models.sd import StableDiffusion
from models.vits import ViT


from datasets.tapvid import TAPVid

def get_args():
    parser = argparse.ArgumentParser("Foundation Models for Point Tracking - Zero Shot")
    parser.add_argument('--tapvid_root', type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, choices=["davis_first", "davis_strided", "rgb_stacking_first", "rgb_stacking_strided", "kinetics_first", "kinetics_strided"], default="davis_first")
    parser.add_argument('--final_feature_size', type=int, nargs=2, default=[32, 32])
    parser.add_argument('--backbone', type=str, choices=["dinov2_s_14",
                                                         "dinov2_b_14", 
                                                         "dinov2_l_14", 
                                                         "dinov2_g_14", 
                                                         "dinov2reg_s_14", 
                                                         "dinov2reg_b_14", 
                                                         "dino_s_16", 
                                                         "dino_s_8",
                                                         "dino_b_16",
                                                         "dino_b_8",
                                                         "mae_b_16",
                                                         "clip_b_16",
                                                         "deit3_b_16",
                                                         "sam_b_16",
                                                         "sd"], default="dinov2_b_14")
    args = parser.parse_args()

    # Extra settings for the models
    args.patch_size = 8 if args.backbone == "sd" else int(args.backbone.split("_")[-1])
    args.input_size = [args.patch_size * fs for fs in args.final_feature_size]

    assert os.path.exists(args.tapvid_root), f"Path {args.tapvid_root} does not exist."

    return args


def main_worker(args):
    cudnn.benchmark = False
    cudnn.deterministic = True

    # === Set up the dataset ===
    dataset = TAPVid(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=5, drop_last=False, pin_memory=True)
    evaluator = Evaluator(zero_shot=True)
    queried_first = "first" in args.eval_dataset

    # === Set up the model ===
    args.gpu = 0        # Single GPU

    if args.backbone == "sd":
        model = StableDiffusion(args, up_block_index=2, repeat_times=8).to(args.gpu)
    else:
        model = ViT(args, frozen=True).to(args.gpu)
    model.eval()
    
    # === Evaluation loop ===
    for j, (video, trajectory, visibility, query_points_i) in enumerate(dataloader):
        query_points_i = query_points_i.cuda(non_blocking=True)      # (1, N, 3)
        trajectory = trajectory.cuda(non_blocking=True)              # (1, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)              # (1, T, N)
        video = video.cuda(non_blocking=True)                        # (1, T, 3, H, W)
        B, T, N, _ = trajectory.shape
        _, _, _, H, W = video.shape
        device = video.device
        
        queries = query_points_i.clone().float()
        queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(device)

    
        tokens, query_tokens = model(video, queries.clone())         # (1, T, D, Hf, Wf), (1, N, D)
        Hf, Wf = tokens.shape[-2:]
        
        cost_volumes = calculate_cost_volume(tokens, query_tokens)   # (1, T, N, Hf, Wf)

        
        cost_volumes_flat = cost_volumes.view(B, T, N, -1)           # (1, T, N, Hf * Wf)
        max_indices = torch.argmax(cost_volumes_flat, dim=-1)        # (1, T, N)
        max_x = max_indices % Wf
        max_y = max_indices // Wf

        # in [Hf, Wf] space, center of the patch
        max_coordinates = torch.stack((max_x, max_y), dim=-1) + 0.5 # (1, T, N, 2)

        # scale to [H, W] space
        max_coordinates[:, :, :, 0] *= (W / Wf)
        max_coordinates[:, :, :, 1] *= (H / Hf)

        # Evaluation - torch to numpy conversion 
        query_points = query_points_i.clone().cpu().numpy()
        gt_tracks = trajectory.clone().permute(0, 2, 1, 3).cpu().numpy()                          # (1, N, T, 2)
        gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()        # (1, N, T)
        pred_tracks = max_coordinates.permute(0, 2, 1, 3).cpu().numpy()                           # (1, N, T, 2)
        pred_occluded = torch.zeros_like(visibility).permute(0, 2, 1).cpu().numpy()               # (1, N, T)

        out_metrics = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first" if queried_first else "strided")
        evaluator.update(out_metrics)

        # break

    evaluator.report()

if __name__ == '__main__':
    args = get_args()
    main_worker(args)
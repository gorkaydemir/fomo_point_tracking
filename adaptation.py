import os
import numpy as np
from tqdm import tqdm
import time
import datetime
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import random


from utils.dist_utils import fix_random_seeds, save_on_master, init_distributed_mode

from read_args import get_args
from datasets.movi_f import get_queries
from utils.eval_utils import compute_tapvid_metrics, Evaluator

from utils.utils import get_dataloaders, get_scheduler, save_on_master

from models.ViTAP import ViTAP
from models.loss import Tracking_Criterion


def train(args, train_dataloader, model, optimizer, lr_scheduler):
    model.train()
    total_loss = 0

    loss_fn = Tracking_Criterion()
    train_dataloader = tqdm(train_dataloader, disable=args.gpu != 0)
    for i, (video, tracks, visibility, gotit) in enumerate(train_dataloader):
        video = video.cuda(non_blocking=True)               # (B, T, C, H, W)
        tracks = tracks.cuda(non_blocking=True)             # (B, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)     # (B, T, N)
        gotit = gotit.cuda(non_blocking=True)[:, 0].bool()  # (B)

        if gotit.sum() == 0:
            continue
        tracks = tracks[gotit]
        visibility = visibility[gotit]
        video = video[gotit]

        queries = get_queries(tracks, visibility)           # (B, N, 3)

        occ_logit, pred_coord = model(video, queries)       # (B * T * N, 1), (B * T * N, 2)
        coord_loss, occ_loss = loss_fn(pred_coord, tracks, occ_logit, visibility)

        loss = coord_loss + occ_loss
        total_loss += loss.item()

        # Display
        lr = optimizer.param_groups[0]["lr"]
        train_dataloader.set_description(f"lr: {lr:.5f} | coord_loss: {coord_loss.item():.4f}, occ_loss: {occ_loss.item():.4f}")

        # Backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / len(train_dataloader)

@torch.no_grad()
def evaluate(args, val_dataloader, model):
    model.eval()

    evaluator = Evaluator(zero_shot=False)
    queried_first = "first" in args.eval_dataset

    for j, (video, trajectory, visibility, query_points_i) in enumerate(tqdm(val_dataloader, disable=args.gpu != 0)):
        query_points_i = query_points_i.cuda(non_blocking=True)      # (1, N, 3)
        trajectory = trajectory.cuda(non_blocking=True)              # (1, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)              # (1, T, N)
        video = video.cuda(non_blocking=True)                    # (1, T, 3, H, W)
        B, T, N, _ = trajectory.shape
        _, _, _, H, W = video.shape
        device = video.device

        queries = query_points_i.clone().float()
        queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(device)

        occ_logit, pred_coord = model.module(video, queries)    # (1 * T * N, 1), (1 * T * N, 2)
        pred_trj = pred_coord.view(B, T, N, 2)
        pred_vsb = torch.sigmoid(occ_logit).view(B, T, N) < 0.5

        # === Metrics ===
        pred_trajectory, pred_visibility = pred_trj, pred_vsb
        traj = trajectory.clone()
        query_points = query_points_i.clone().cpu().numpy()
        gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
        gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_occluded = torch.logical_not(pred_visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()

        out_metrics = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first" if queried_first else "strided")
        evaluator.update(out_metrics, verbose=False)

    delta_avg = sum(evaluator.delta_avg) / len(evaluator.delta_avg)
    aj = sum(evaluator.aj) / len(evaluator.aj)
    oa = sum(evaluator.oa) / len(evaluator.oa)
    results = {"delta_avg": delta_avg, "aj": aj, "oa": oa}

    return results


def main_worker(args):
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    cudnn.benchmark = False

    train_dataloader, val_dataloader = get_dataloaders(args)
    print(f"Total number of iterations: {(len(train_dataloader)) * args.epoch_num / 1000:.1f}K")

    model = ViTAP(args).to(args.gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = get_scheduler(args, optimizer, train_dataloader)

    print()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1000:.2f}K")
    print(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000:.2f}K")
    print()

    dist.barrier()
    for epoch in range(args.epoch_num):
        print(f"=== === Epoch {epoch} === ===")
        train_dataloader.sampler.set_epoch(epoch)

        # Train
        loss = train(args, train_dataloader, model, optimizer, lr_scheduler)
        
        if args.gpu == 0:
            print(f"Loss: {loss:.4f}\n")

            results = evaluate(args, val_dataloader, model)
            print(f"AJ: {results['aj']:.2f}, Delta Avg: {results['delta_avg']:.2f}, OA: {results['oa']:.2f}")

            # === Save Model ===
            if args.model_save_path is not None:
                save_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "args": args,
                }
                save_on_master(save_dict, os.path.join(args.model_save_path, "checkpoint.pt"))

        dist.barrier()

    dist.destroy_process_group()



if __name__ == '__main__':
    args = get_args()
    main_worker(args)
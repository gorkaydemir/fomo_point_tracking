import os

import torch
import torch.nn.functional as F
import torch.distributed as dist

import shutil

from datasets.movi_f import Movi_F
from datasets.tapvid import TAPVid


# === Point tracking Utils ===
def regional_soft_argmax(cost_volumes, delta_treshold=16):
    # :args cost_volumes: (B, T, N, H, W)
    #
    # :return points: (B, T, N, 2)

    B, T, N, H, W = cost_volumes.shape
    B_prime = B * T * N
    device = cost_volumes.device
    cost_volumes = cost_volumes.view(B_prime, H, W)

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.cat((xx.unsqueeze(dim=-1), yy.unsqueeze(dim=-1)), dim=-1)  # (H, W, 2)
    grid = grid.to(device)
    grid = grid.unsqueeze(dim=0).expand(B_prime, -1, -1, -1)                # (B', H, W, 2)
    grid = grid + 0.5

    flat_indices = torch.argmax(cost_volumes.view(B_prime, -1), dim=1)      # (B')
    max_y = flat_indices // W                                               # (B')             
    max_x = flat_indices % W                                                # (B')
    max_locations = torch.stack((max_x, max_y), dim=1) + 0.5                # (B', 2)

    spatial_mask = (grid - max_locations.unsqueeze(dim=1).unsqueeze(dim=1)) ** 2    # (B', H, W, 2)
    spatial_mask = torch.sum(spatial_mask, dim=-1) < delta_treshold ** 2            # (B', H, W)

    weights = cost_volumes * spatial_mask                                        # (B', H, W)
    weights = weights / weights.sum(dim=-1).sum(dim=-1).view(B_prime, 1, 1)      # (B', H, W)
    weights = weights.unsqueeze(dim=-1) * grid                                   # (B', H, W, 2)
    points = weights.sum(dim=1).sum(dim=1)                                       # (B', 2)

    points = points.view(B, T, N, 2)
    
    return points


def calculate_cost_volume(tokens, query_tokens):
    # :args tokens: (B, T, D, H, W)
    # :args query_tokens: (B, N, D)
    #
    # :return cost_volumes: (B, T, N, H, W)

    B, T, D, H, W = tokens.shape
    B, N, D = query_tokens.shape
    device = tokens.device

    query_tokens = F.normalize(query_tokens, dim=-1)
    tokens = F.normalize(tokens, dim=2)
    
    cost_volumes = torch.zeros(B, T, N, H, W, device=device)
    for t in range(T):
        tokens_t = tokens[:, t].view(B, D, H * W)                            # (B, D, P)
                
        cost_volume = torch.einsum("bnd,bdp->bnp", query_tokens, tokens_t)   # (B, N, P)
        cost_volume = cost_volume.view(B, N, H, W)
        
        cost_volumes[:, t] = cost_volume

    return cost_volumes


def get_dataloaders(args):
    train_dataset = Movi_F(args)
    val_dataset = TAPVid(args)

    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=args.gpus, rank=args.gpu, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.bs // args.gpus,
        num_workers=5,      # cpu per gpu
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
        batch_size=1,
        shuffle=False, 
        num_workers=5, 
        drop_last=False,
        pin_memory=True)

    return train_dataloader, val_dataloader


def get_scheduler(args, optimizer, train_loader):
    T_max = len(train_loader) * args.epoch_num
    warmup_steps = int(T_max * 0.01)
    steps = T_max - warmup_steps

    linear_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, total_iters=warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[warmup_steps])

    return scheduler


# === Distributed Utils ===
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
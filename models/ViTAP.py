import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from models.vits import ViT

from utils.utils import regional_soft_argmax, calculate_cost_volume

class ViTAP(nn.Module):
    def __init__(self, args):
        super().__init__()


        self.f_size = args.final_feature_size

        frozen_backbone = True if args.mode == "probe" else False
        self.vit = ViT(args, frozen=frozen_backbone)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

        # occlusion branch
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.occ_mlp = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        # cordinate branch
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, video, queries):
        # :args video: (B, T, 3, H, W) in range [0, 255]
        # :args queries: (B, N, 3) where 3 is (t, y, x)
        #
        # :return occ_logit: (B * T * N, 1)
        # :return pred_coord: (B * T * N, 2)
        
        H, W = video.shape[-2:]

        features, queries = self.vit(video, queries)                        # (B, T, D, H, W), (B, N, D)

        cost_volumes = calculate_cost_volume(features, queries)            # (B, T, N, Hf, Wf)
        B, T, N, Hf, Wf = cost_volumes.shape
        
        assert Hf == self.f_size[0] and Wf == self.f_size[1]

        cost_volumes = cost_volumes.view(B * T * N, 1, Hf, Wf)

        # Common Branch
        x = F.relu(self.conv1(cost_volumes))                                # (B * T * N, 16, Hf, Wf)

        # === Occlusion Branch ===
        x_occ = F.relu(self.conv2(x))                                       # (B * T * N, 32, Hf//2, Wf//2)
        x_occ = F.adaptive_avg_pool2d(x_occ, (1, 1)).view(-1, 32)           # (B * T * N, 32)
        occ_logit = self.occ_mlp(x_occ)                                     # (B * T * N, 1)

        # === Coordinate Branch ===
        x_coord = self.conv3(x).view(B * T * N, Hf * Wf)                    # (B * T * N, Hf * Wf)
        x_coord = F.softmax(x_coord, dim=-1)                                # (B * T * N, Hf * Wf)

        x_coord = x_coord.view(B, T, N, Hf, Wf)                             # (B, T, N, Hf, Wf)
        pred_coord = regional_soft_argmax(x_coord, delta_treshold=4)        # (B, T, N, 2)
        pred_coord = pred_coord.view(B * T * N, 2)                          # (B * T * N, 2)

        pred_coord[:, 1] *= (H / Hf)
        pred_coord[:, 0] *= (W / Wf)

        return occ_logit, pred_coord



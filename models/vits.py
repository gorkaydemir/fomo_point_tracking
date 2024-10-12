import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import timm
from utils.lora_utils import LoRA_ViT_timm

model_config = {"dinov2_s_14": ["vit_small_patch14_dinov2.lvd142m", 14, 384],
                "dinov2_b_14": ["vit_base_patch14_dinov2.lvd142m", 14, 768],
                "dinov2_l_14": ["vit_large_patch14_dinov2.lvd142m", 14, 1024],
                "dinov2_g_14": ["vit_giant_patch14_dinov2.lvd142m", 14, 1536],
                "dinov2reg_s_14": ["vit_small_patch14_reg4_dinov2.lvd142m", 14, 384],
                "dinov2reg_b_14": ["vit_base_patch14_reg4_dinov2.lvd142m", 14, 768],
                "dino_s_16": ["vit_small_patch16_224.dino", 16, 384],
                "dino_s_8": ["vit_small_patch8_224.dino", 8, 384],
                "dino_b_16": ["vit_base_patch16_224.dino", 16, 768],
                "dino_b_8": ["vit_base_patch8_224.dino", 8, 768],
                "mae_b_16": ["vit_base_patch16_224.mae", 16, 768],
                "clip_b_16": ["vit_base_patch16_clip_224.openai", 16, 768],
                "deit3_b_16": ["deit3_base_patch16_224.fb_in22k_ft_in1k", 16, 768],
                "sam_b_16": ["samvit_base_patch16.sa1b", 16, 256]}

class ViT(nn.Module):
    def __init__(self, args, frozen=True):
        super().__init__()

        self.frozen = frozen

        self.model_name, self.patch_size, self.encoder_dim = model_config[args.backbone]

        self.size = args.input_size
        assert self.size[0] % self.patch_size == 0 and self.size[1] % self.patch_size == 0
        self.fsize = [s // self.patch_size for s in self.size]

        if args.backbone == "sam_b_16":
            model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        else:
            model = timm.create_model(self.model_name, img_size=self.size, pretrained=True, num_classes=0)
            
        self.normalization = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        if self.frozen:
            self.model = model
            for param in self.model.parameters():
                param.requires_grad = False

        else:
            assert "dinov2" in args.backbone, f"Only dinov2 models are supported for learnable backbones."
            self.model = LoRA_ViT_timm(model, r=args.lora_rank)


    def sample_points(self, tokens, queries):
        # :args tokens: (B, T, D, H, W)
        # :args queries: (B, N, 3) where 3 is (t, x, y)
        #
        # :return queries: (B, N, C)

        B, N, _ = queries.shape
        D = tokens.shape[2]
        device = tokens.device

        query_features = torch.zeros(B, N, D, device=device)        # (B, N, D)
        query_points_reshaped = queries.view(-1, 3)                 # (B * N, 3)
        t, x, y = query_points_reshaped[:, 0].long(), query_points_reshaped[:, 1], query_points_reshaped[:, 2]          # (B * N)

        source_frame_f = tokens[torch.arange(B).repeat_interleave(N), t].reshape(-1, D, tokens.size(-2), tokens.size(-1))  # (B * N, C, H, W)
        x_grid = (x / self.size[1]) * 2 - 1
        y_grid = (y / self.size[0]) * 2 - 1

        grid = torch.stack([x_grid, y_grid], dim=-1).view(B * N, 1, 1, 2).to(device)
        sampled = F.grid_sample(source_frame_f, grid, mode='bilinear', padding_mode='border', align_corners=False)
        query_features.view(-1, D)[:] = sampled.reshape(-1, D)      # (B * N, C)
        query_features = query_features.view(B, N, D)

        return query_features
        

    def forward(self, video, queries):
        # :args video: (B, T, 3, H, W) in range [0, 255]
        # :args queries: (B, N, 3) where 3 is (t, y, x)
        #
        # :return tokens: (B, T, D, H, W)
        # :return queries: (B, N, D)
        
        B, T, C, H_in, W_in = video.shape
        B, N, _ = queries.shape

        video_flat = video.view(B * T, C, H_in, W_in) / 255.0             # (B * T, 3, H_in, W_in), to [0, 1]
        video_flat = F.interpolate(video_flat, size=self.size, mode="bilinear", align_corners=False)
        video_flat = self.normalization(video_flat)         # to [-1, 1]

        queries[:, :, 2] = (queries[:, :, 2] / H_in) * self.size[0]
        queries[:, :, 1] = (queries[:, :, 1] / W_in) * self.size[1]

        # ==== Accumulate tokens from the video for efficiency ====
        
        acc = []
        split_size = 32         # arbitrary slot size
        start_idx = 0
        while start_idx < video_flat.shape[0]:
            if self.frozen:
                if "sam" in self.model_name:
                    tokens_tmp = self.model.forward_features(video_flat[start_idx: start_idx + split_size])                  # (split_size, 256, Hf, Wf)
                    tokens_tmp = tokens_tmp.view(tokens_tmp.size(0), tokens_tmp.size(1), -1)
                    tokens_tmp = tokens_tmp.permute(0, 2, 1)

                elif "reg" in self.model_name:
                    tokens_tmp = self.model.forward_features(video_flat[start_idx: start_idx + split_size])[:, 5:]           # (split_size, P, D)

                else:
                    tokens_tmp = self.model.forward_features(video_flat[start_idx: start_idx + split_size])[:, 1:]           # (split_size, P, D)

            else:
                tokens_tmp = self.model(video_flat[start_idx: start_idx + split_size])           # (split_size, P, D)

            acc.append(tokens_tmp)
            start_idx += split_size
        
        tokens = torch.cat(acc, dim=0)
        assert tokens.shape[0] == B * T

        # ==== Sample points from the tokens ====
        _, P, D = tokens.shape
        tokens = tokens.view(B, T, self.fsize[0], self.fsize[1], D)       # (B, T, H, W, D)
        tokens = tokens.permute(0, 1, 4, 2, 3)                            # (B, T, D, H, W)

        queries = self.sample_points(tokens, queries)                     # (B, N, D)
        
        return tokens, queries
        
        
        

        



        
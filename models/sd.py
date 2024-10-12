from transformers import CLIPTextModel, CLIPTokenizer, logging

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random


import numpy as np
import math

import os

# Below is adapted from https://github.com/aliasgharkhani/SLiMe/blob/main/src/stable_difusion.py
class StableDiffusion(nn.Module):
    def __init__(
        self,
        args,
        up_block_index=2, # starts from 0
        repeat_times=4,
        sd_version="1.5",
    ):
        super().__init__()

        self.sd_version = sd_version
        if self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == "1.4":
            model_key = "CompVis/stable-diffusion-v1-4"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        self.size = args.input_size
        self.fsize = [s // 8 for s in self.size]
        
        self.repeat_times = repeat_times
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet")

        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False

        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()
        self.scheduler = DDIMScheduler.from_config(model_key, subfolder="scheduler")
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.020)
        self.max_step = int(self.num_train_timesteps * 0.980)

        self.alphas = self.scheduler.alphas_cumprod  # for convenience                
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unet.to(self.device)
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        
        
        assert up_block_index in range(4)
        self.noise = None
        self.features = []
        
        self.unet.up_blocks[up_block_index].register_forward_hook(self.get_features())

        self.text_embeds = self.get_text_embeds("").to(self.device)                  # (1, 77, 768)
        
    def get_features(self):
        def hook(module, input, output):
            self.features.append(output.detach())
        return hook

    def noise_latents(self, latents, t):
        B = latents.shape[0]
        assert t < self.scheduler.config.num_train_timesteps

        timesteps = torch.ones(B, device=latents.device) * t
        timesteps = timesteps.long()

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        return noisy_latents, noise
    
    def encode(self, frames):
        # :arg frames: tensor of shape (B, 3, H, W)
        #
        # :return repeated_latents: tensor of shape (B * repeat_times, 3, H, W)
        
        latent_dist = self.vae.encode(frames.to(self.vae.dtype)).latent_dist
        latents_repeated = []
        for _ in range(self.repeat_times):
            latents = latent_dist.sample() * 0.18215          # (B, 4, 64, 64)
            latents_repeated.append(latents)
            
        repeated_latents = torch.cat(latents_repeated, dim=0)  # (B * repeat_times, 4, 64, 64)

        # Reshape to interleave the samples as [0_0, 0_1, ..., 0_N, ..., B_0,... B_N]
        repeated_latents = repeated_latents.view(self.repeat_times, -1, *repeated_latents.shape[1:]).transpose(0, 1).contiguous()
        repeated_latents = repeated_latents.view(-1, *repeated_latents.shape[2:])
        
        return repeated_latents

    def get_text_embeds(self, prompt):
        # return: text features (1, 77, 768)
        
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.set_grad_enabled(False):
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return text_embeddings 

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

    @torch.no_grad()
    def forward(self, video, queries, t=51):
        # :args video: (B, T, 3, H, W) in range [0, 255]
        # :args queries: (B, N, 3) where 3 is (t, y, x)
        #
        # :return tokens: (B, T, D, H, W)
        # :return queries: (B, N, D)

        B, T, C, H_in, W_in = video.shape
        B, N, _ = queries.shape

        video_flat = video.view(B * T, C, H_in, W_in) / 255.0                                           # (B * T, 3, H_in, W_in), to [0, 1]
        video_flat = F.interpolate(video_flat, size=self.size, mode="bilinear", align_corners=False)
        video_flat = (video_flat * 2.0) - 1.0

        queries[:, :, 2] = (queries[:, :, 2] / H_in) * self.size[0]
        queries[:, :, 1] = (queries[:, :, 1] / W_in) * self.size[1]
        
                
        text_embeds = self.text_embeds.expand(B * T * self.repeat_times, -1, -1)       # (B * T * repeat_times, 77, 768)
        latents = []
        split_size = 32
        start_idx = 0
        while start_idx < (video_flat.shape[0]):
            latents_i = self.encode(video_flat[start_idx:start_idx+split_size])
            latents.append(latents_i)
            start_idx += split_size
        latents = torch.cat(latents, dim=0)                                              # (B * T * repeat_times, 4, H_in, W_in)
        
        noisy_latents, _ = self.noise_latents(latents, t)                              # (B * T * repeat_times, 4, H_in, W_in)

        acc = []
        split_size = 32
        start_idx = 0
        while start_idx < (video_flat.shape[0] * self.repeat_times):
            _ = self.unet(noisy_latents[start_idx:start_idx+split_size], torch.tensor(t).to(self.device), text_embeds[start_idx:start_idx+split_size])
            f = self.features.pop()                                                        # (B * T * repeat_times, D, Hf, Wf)
            acc.append(f)
            start_idx += split_size

        f = torch.cat(acc, dim=0)
        D = f.shape[1]
        
        f = f.view(B, T, self.repeat_times, D, self.fsize[0], self.fsize[1])           # (B, T, repeat_times, D, 64, 64)
        f = torch.mean(f, dim=2)                                                       # (B, T, D, 64, 64)


        queries = self.sample_points(f, queries)                                       # (B, N, D)
        
        return f, queries
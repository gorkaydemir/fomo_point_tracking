import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class Tracking_Criterion(nn.Module):
    def __init__(self):
        super().__init__()

        self.huber_loss = nn.HuberLoss(reduction="none", delta=7)
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.lambda_coord = 0.1
        self.lambda_occ = 10

    def forward(self, pred_coord, gt_coord, occ_logit, gt_visibility):
        # :args pred_coord: (B * T * N, 2)
        # :args gt_coord: (B, T, N, 2)
        # :args occ_logit: (B * T * N, 1)
        # :args gt_visibility: (B, T, N)

        B, T, N, _ = gt_coord.shape

        visible_mask = gt_visibility.view(B * T * N, 1).float() # (B * T * N, 1)

        # === Coordinate Loss ===
        gt_coord = gt_coord.view(B * T * N, 2)                  # (B * T * N, 2)
        coord_loss = self.huber_loss(pred_coord, gt_coord)      # (B * T * N, 2)
        coord_loss = coord_loss.sum(dim=-1)                     # (B * T * N)
        coord_loss = coord_loss * visible_mask                  # (B * T * N)
        coord_loss = coord_loss.mean()

        # === Occlusion Loss ===
        gt_occlusions = 1 - gt_visibility.view(B * T * N, 1).float()    # (B * T * N, 1)
        occ_loss = self.bce_with_logits_loss(occ_logit, gt_occlusions)  # (B * T * N, 1)
        occ_loss = occ_loss.squeeze(dim=-1)                             # (B * T * N)
        occ_loss = occ_loss.mean()

        return self.lambda_coord * coord_loss, self.lambda_occ * occ_loss

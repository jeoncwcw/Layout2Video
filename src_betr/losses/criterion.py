import torch
from torch import nn
import torch.nn.functional as F
from .utils import TargetGenerator

class BETRLoss(nn.Module):
    def __init__(self, lambda_fine=2.0, sigma=2.0, heatmap_size=128, input_size=512):
        super(BETRLoss, self).__init__()
        self.heatmap_size, self.input_size = int(heatmap_size), int(input_size)
        self.lambda_fine = lambda_fine
        self.target_gen = TargetGenerator(heatmap_size=heatmap_size, sigma=sigma)
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def forward(self, preds, batch):
        """
        preds['center coords']: [B, 1, 2] (0-511 scale)
        batch['gt_center']: [B, 2] (0-1 scale)
        batch['padding_mask']: [B, 512, 512] (bool, True for padding)
        """
        device = preds["center heatmap"].device

        raw_mask = batch["padding_mask"].unsqueeze(1).float()  # [B, 1, 512, 512]
        valid_mask = 1.0 - F.interpolate(raw_mask, size=(self.heatmap_size, self.heatmap_size), mode="nearest")
        valid_mask = valid_mask.to(device)

        weight_map = self.target_gen.generate_heatmap(batch["gt_center"], device)
        weight_map = weight_map * valid_mask  # Mask out padding areas

        # -- [Center Loss] --
        # L_coarse
        coarse_diff = self.smooth_l1(preds['center heatmap'], weight_map)
        loss_coarse = (coarse_diff * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        # L_fine
        pred_center_norm = preds['center coords'].squeeze(1) / float(self.input_size)
        loss_fine = F.smooth_l1_loss(pred_center_norm, batch['gt_center'])

        loss_center = loss_coarse + self.lambda_fine * loss_fine

        # -- [Weighted Dense Losses] --
        gt_offsets_map = batch['gt_offsets_3d'].view(-1, 16, 1, 1).expand(-1, -1, self.heatmap_size, self.heatmap_size)  # [B, 16, 128, 128]
        loss_offset = self._weighted_loss(preds['bb8 offsets'], gt_offsets_map, weight_map)

        gt_depth_map = batch['gt_center_depth'].view(-1, 1, 1, 1).expand(-1, 1, self.heatmap_size, self.heatmap_size)  # [B, 1, 128, 128]
        loss_depth = self._weighted_loss(preds['center depth'], gt_depth_map, weight_map)

        gt_depth_offsets_map = batch['gt_depth_offsets'].view(-1, 8, 1, 1).expand(-1, -1, self.heatmap_size, self.heatmap_size)  # [B, 8, 128, 128]
        loss_depth_offset = self._weighted_loss(preds['bb8 depth offsets'], gt_depth_offsets_map, weight_map)

        total_loss = loss_center + loss_offset + loss_depth + loss_depth_offset
        return {
            "total_loss": total_loss,
            "loss_center": loss_center,
            "loss_offset": loss_offset,
            "loss_depth": loss_depth,
            "loss_depth_offset": loss_depth_offset,
        }
    def _weighted_loss(self, pred, target, weight_map):
        loss = self.smooth_l1(pred, target)
        weighted_loss = (loss * weight_map).sum() / (weight_map.sum() * pred.shape[1]+ 1e-8)
        return weighted_loss
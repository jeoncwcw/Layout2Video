import torch
from torch import nn
import torch.nn.functional as F
from .utils import TargetGenerator

# MEAN = {"center": 0.518, "bb8_offset": 0.0, "center_depth": 6.511, "bb8_depth_offset": -0.007}
# STD = {"center": 0.159, "bb8_offset": 0.084, "center_depth": 0.968, "bb8_depth_offset": 0.120}

class BETRLoss(nn.Module):
    def __init__(self, start_lambda_fine=5.0, end_lambda_fine=10.0, loss_depth = 1.0,
                 heatmap_size=128, input_size=512,
                 threshold=0.01, start_peak=200.0, end_peak=20.0,
                 total_epochs=100):
        super(BETRLoss, self).__init__()
        self.heatmap_size, self.input_size = int(heatmap_size), int(input_size)
        self.start_lambda_fine = start_lambda_fine
        self.end_lambda_fine = end_lambda_fine
        self.loss_depth = loss_depth
        self.target_gen = TargetGenerator(heatmap_size=heatmap_size) 
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.threshold = threshold
        self.start_peak = start_peak
        self.end_peak = end_peak
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def get_current_peak_weight(self):
        progress = self.current_epoch / self.total_epochs
        return self.start_peak - progress * (self.start_peak - self.end_peak)
    
    def get_current_lambda_fine(self):
        progress = self.current_epoch / self.total_epochs
        return self.start_lambda_fine + progress * (self.end_lambda_fine - self.start_lambda_fine)
    
    def forward(self, preds, batch):
        """
        preds['corner coords']: [B, 8, 2] (0-511 scale)
        preds['corner heatmaps']: [B, 8, 128, 128]
        preds['corner depths']: [B, 8, 128, 128]
        batch['gt_corners']: [B, 8, 2] (0-1 scale)
        batch['padding_mask']: [B, 512, 512] (bool, True for padding)
        """
        device = preds["corner heatmaps"].device
        raw_mask = batch["padding_mask"].unsqueeze(1).float()  # [B, 1, 512, 512]
        valid_mask = 1.0 - F.interpolate(raw_mask, size=(self.heatmap_size, self.heatmap_size), mode="nearest") # [B, 1, 128, 128]
        valid_mask = valid_mask.to(device)
        gt_corners = batch['gt_corners']  # [B, 8, 2]

        weight_map = self.target_gen.generate_heatmap(gt_corners, device) # [B, 8, 128, 128]
        weight_map = weight_map * valid_mask  # Mask out padding areas

       
        pred_confidence_map = preds['corner heatmaps']  # [B, 8, 128, 128]
        pred_confidence_map = pred_confidence_map * valid_mask  # Mask out padding areas
        # -- [Corner Loss] --
        # L_coarse
         # TODO: change code pos_weight to be tuned by cfg file
        B, C, _, _ = preds['corner heatmaps'].shape
        pos_weight = torch.where(weight_map > self.threshold, self.get_current_peak_weight(), 1.0)
        coarse_diff = self.smooth_l1(preds['corner heatmaps'], weight_map)
        loss_coarse = (coarse_diff * pos_weight * valid_mask).sum() / (weight_map.sum() + 1e-8)
        # L_fine
        pred_corner_norm = preds['corner coords'] / float(self.input_size)
        loss_fine = F.smooth_l1_loss(pred_corner_norm, batch['gt_corners'], reduction='mean')

        loss_corners = loss_coarse + self.get_current_lambda_fine() * loss_fine

        # -- [Weighted Dense Losses] --
        gt_corner_depths_map = batch['gt_depths'].view(-1, 8, 1, 1).expand(-1, 8, self.heatmap_size, self.heatmap_size)  # [B, 8, 128, 128]
        loss_depths = self._weighted_loss(preds['corner depths'], gt_corner_depths_map, pred_confidence_map)

        total_loss = loss_corners + self.loss_depth * loss_depths
        return {
            "total_loss": total_loss,
            "loss_corners": loss_corners,
            "loss_depths": loss_depths,
        }
    def _weighted_loss(self, pred, target, weight_map):
        loss = self.smooth_l1(pred, target)
        weighted_loss = (loss * weight_map).sum() / (weight_map.sum() + 1e-8)
        return weighted_loss
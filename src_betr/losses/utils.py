import torch
import numpy as np

class TargetGenerator:
    def __init__(self, heatmap_size=128, sigma=2.0):
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        y, x = torch.meshgrid(
            torch.arange(heatmap_size),
            torch.arange(heatmap_size),
            indexing='ij'
        )
        self.grid_y = y.float()
        self.grid_x = x.float()

    def generate_heatmap(self, gt_corners, device):
        """       
        :param gt_corners: [B, 8, 2] Normalized coordinates of corners (0-1)
        :param device: torch device
        Returns: [B, 8, 128, 128] weight map
        """
        B = gt_corners.shape[0]
        H = W = self.heatmap_size
        
        gt_x = (gt_corners[:, :, 0] * (W - 1)).unsqueeze(-1) # [B, 8, 1]
        gt_y = (gt_corners[:, :, 1] * (H - 1)).unsqueeze(-1) # [B, 8, 1]
        grid_x = self.grid_x.to(device).view(1, 1, H, W) # [1, 1, 128, 128]
        grid_y = self.grid_y.to(device).view(1, 1, H, W) # [1, 1, 128, 128]
        
        dist_sq = (grid_x - gt_x.unsqueeze(-1))**2 + (grid_y - gt_y.unsqueeze(-1))**2  # [B, 8, 128, 128]
        weight_map = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        return weight_map # [B, 8, 128, 128]
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
        self.register_buffer('grid_x', x.float())
        self.register_buffer('grid_y', y.float())

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def generate_heatmap(self, gt_centers, device):
        """       
        :param gt_center_norm: [B, 2] Normalized coordinates of center
        Returns: [B, 1, 128, 128] weight map
        """
        B = gt_centers.shape[0]
        H = W = self.heatmap_size
        
        gt_x = gt_centers[:, 0:1] * (W - 1)
        gt_y = gt_centers[:, 1:2] * (H - 1)

        dx = self.grid_x.to(device).unsqueeze(0) - gt_x.unsqueeze(-1)
        dy = self.grid_y.to(device).unsqueeze(0) - gt_y.unsqueeze(-1)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        
        weight_map = torch.exp(-dist / (2 * self.sigma ** 2))
        return weight_map.unsqueeze(1) # [B, 1, 128, 128]
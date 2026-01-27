import torch
import torch.nn.functional as F


class CornerGeometryMetric:
    def __init__(self, device="cuda"):
        self.device = device
        self.reset()
        
    def reset(self):
        self.total_uv_dist = 0.0
        self.total_depth_diff = 0.0
        self.total_samples = 0
        # Corner-wise stats
        self.corner_uv_dists = torch.zeros(8, device=self.device)
        self.corner_depth_diffs = torch.zeros(8, device=self.device)
        
    def sample_depths(self, depth_maps, coords):
        """
        depth_maps: [B, 8, H, W]
        coords: [B, 8, 2] - (u,v) coordinates
        """
        B, num_corners, H, W = depth_maps.shape
        
        # normalize coords to [-1, 1] for grid_sample
        norm_coords = coords.clone()
        norm_coords[..., 0] = 2.0 * (norm_coords[..., 0] / (W - 1)) - 1.0 # u
        norm_coords[..., 1] = 2.0 * (norm_coords[..., 1] / (H - 1)) - 1.0 # v
        
        flat_depth_maps = depth_maps.view(B * num_corners, 1, H, W)
        flat_grid = norm_coords.view(B * num_corners, 1, 1, 2)
        
        sample_depths = F.grid_sample(flat_depth_maps, flat_grid, mode="bilinear", align_corners=True)
        return sample_depths.view(B, num_corners)
    
    def update(self, outputs, batch):
        """
        outputs: model prediction dictionary
        batch: GT batch
        """
        # Get predictions and GT
        pred_uv = outputs['corner coords'].detach() # [B, 8, 2]
        pred_depth_maps = outputs['corner depths'].detach() # [B, 8, 128, 128]
        gt_uv = batch['gt_corners'] * 512.0 # [B, 8, 2] - scale to 0-511
        gt_d = torch.exp(batch['gt_depths']) # [B, 8] - in meters
        
        pred_uv_128 = pred_uv / 4.0 
        pred_log_d = self.sample_depths(pred_depth_maps, pred_uv_128)
        pred_d = torch.exp(pred_log_d)
        
        # Carculate errors (uv: euclidean distance in pixels, depth: absolute error in meters)
        uv_dist = torch.norm(pred_uv - gt_uv, p=2, dim=-1) # [B, 8]
        depth_diff = torch.abs(pred_d - gt_d) # [B, 8]
        
        batch_size = pred_uv.shape[0]
        self.total_uv_dist += uv_dist.sum().item()
        self.total_depth_diff += depth_diff.sum().item()
        self.total_samples += batch_size
        
    def compute(self):
        if self.total_samples == 0:
            return 0.0, 0.0
    
        total_corners = self.total_samples * 8
        avg_uv_error = self.total_uv_dist / total_corners
        avg_depth_error = self.total_depth_diff / total_corners
        return avg_uv_error, avg_depth_error, avg_uv_error + avg_depth_error * 5.0
    
    def get_result_dict(self):
        avg_uv, avg_d, mixed_score = self.compute()
        return {
            "avg_uv_error": avg_uv,
            "avg_depth_error": avg_d,
            "mixed_score": mixed_score,
        }
        
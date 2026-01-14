import torch
from torch import nn

class BETRv2Loss(nn.Module):
    def __init__(self, center=1.0, offset=1.0, depth=1.0, d_offset=1.0):
        super().__init__()
        self.w = {
            'center': center,
            'offset': offset,
            'depth': depth,
            'd_offset': d_offset
        }
        self.loss_fn = nn.SmoothL1Loss(reduction='mean')

    def forward(self, preds, batch):
        # preds: (B, 27) MLP Heads output
        # batch['gt_...']: Normalized values from the dataloader
        
        # 1. Value slicing (separating predictions)
        p_center   = preds[:, 0:2]    # [B, 2]
        p_offsets  = preds[:, 2:18]   # [B, 16]
        p_depth    = preds[:, 18:19]  # [B, 1]
        p_d_offset = preds[:, 19:27]  # [B, 8]

        # 2. Calculate individual losses (recommended to calculate on normalized values)
        loss_center   = self.loss_fn(p_center, batch['gt_center'])
        loss_offset   = self.loss_fn(p_offsets, batch['gt_offsets_3d'])
        loss_depth    = self.loss_fn(p_depth, batch['gt_center_depth'].view(-1, 1))
        loss_d_offset = self.loss_fn(p_d_offset, batch['gt_depth_offsets'])

        # 3. Weighted sum calculation
        total_loss = (self.w['center'] * loss_center + 
                      self.w['offset'] * loss_offset + 
                      self.w['depth'] * loss_depth + 
                      self.w['d_offset'] * loss_d_offset)

        return {
            "total_loss": total_loss,
            "loss_center": loss_center,
            "loss_offset": loss_offset,
            "loss_depth": loss_depth,
            "loss_depth_offset": loss_d_offset
        }
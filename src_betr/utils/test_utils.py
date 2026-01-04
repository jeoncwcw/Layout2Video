import numpy as np
import cv2
from pathlib import Path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def inverse_depth_norm(norm_inv_depth, min_depth=0.1, max_depth=3000.0):
    inv_min = 1.0 / max_depth
    inv_max = 1.0 / min_depth
    inv_depth = norm_inv_depth * (inv_max - inv_min) + inv_min
    depth = 1.0 / (inv_depth + 1e-8)
    return depth

def visualization(small_batch, outputs, out_dir: Path):
    pred_center = outputs['center coords'].squeeze(1).cpu().numpy() # [B, 2] # unnormalized, dino scale
    pred_offsets = outputs['bb8 offsets'].cpu().numpy() # [B, 16, H, W] # normalized, feature scale (1/4 dino)
    for i in range(4):
        # Convert tensor to image
        image = small_batch['image_dino'][i].permute(1,2,0).cpu().numpy()
        mean = np.array(IMAGENET_MEAN).reshape(1,1,3)
        std = np.array(IMAGENET_STD).reshape(1,1,3)
        image = (image * std + mean) * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # load gt, predicted corners
        corners_3d = small_batch['gt_corners_3d'][i].cpu().numpy() * 512
        heatmap_center_x, heatmap_center_y = int(np.clip(round(pred_center[i][0]/4), 0, 127)), int(np.clip(round(pred_center[i][1]/4), 0, 127))
        pred_offset = pred_offsets[i,:,heatmap_center_y, heatmap_center_x].reshape(8, 2) # [8, 2]
        pred_corners = pred_offset * 512 + pred_center[i][None, :] # unnormalize and to dino scale
        # Draw GT, Predicted corners
        for j in range(8):
            gt_u, gt_v = int(round(corners_3d[j][0])), int(round(corners_3d[j][1]))
            pd_u, pd_v = int(round(pred_corners[j][0])), int(round(pred_corners[j][1]))
            cv2.circle(image, (gt_u, gt_v), 5, (0,255,0), -1) # Green for GT
            cv2.circle(image, (pd_u, pd_v), 5, (0,0,255), -1) # Red for Pred
        cv2.imwrite(str(out_dir / f"vis_{i:03d}.png"), image)
    print(f"Visualization images saved to: {out_dir}")
    
def comparing_depth(small_batch, outputs):
    pred_center_coords = outputs['center coords'].squeeze(1).cpu().numpy()
    pred_center_depth = outputs['center depth'].cpu().numpy() # [B, 1, H, W]
    pred_depth_offsets = outputs['bb8 depth offsets'].cpu().numpy() # [B, 8, H, W]
    gt_center_depth = small_batch['gt_metric_depth'].cpu() # [B], scalar value
    gt_depth_offsets = small_batch['gt_offsets_3d'].cpu().numpy() # [B, 8]
    for i in range(4):
        gt_depths = gt_center_depth[i].item() * np.exp(gt_depth_offsets[i])  # [8]
        heatmap_center_x, heatmap_center_y = int(np.clip(round(pred_center_coords[i][0]/4), 0, 127)), int(np.clip(round(pred_center_coords[i][1]/4), 0, 127))
        pred_offset = pred_depth_offsets[i,:,heatmap_center_y, heatmap_center_x] # [8]
        pred_center = pred_center_depth[i,0,heatmap_center_y, heatmap_center_x].item()
        pred_center = inverse_depth_norm(pred_center, min_depth=0.1, max_depth=3000.0)
        pred_depths = pred_center * np.exp(pred_offset)  # [8]
        print(f"Sample #{i}:")
        print(f"{'Idx':<4} | {'GT Depth (m)':<15} | {'Pred Depth (m)':<15}")
        print("-"*40)
        for j in range(8):
            gt_z = gt_depths[j].item()
            pd_z = pred_depths[j].item()
            print(f"#{j:<3} | {gt_z:<15.2f} | {pd_z:<15.2f}")
        print("="*40 + "\n")
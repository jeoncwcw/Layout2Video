import numpy as np
import cv2
from pathlib import Path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MEAN = {"center": 0.497, "bb8_offset": 0.0, "center_depth": 6.181, "bb8_depth_offset": -0.009}
STD = {"center": 0.144, "bb8_offset": 0.097, "center_depth": 1.002, "bb8_depth_offset": 0.133}

def visualization(small_batch, outputs, out_dir: Path):
    pred_center = outputs['center coords'].squeeze(1).cpu().numpy() # [B, 2] # unnormalized, dino scale
    pred_offsets = outputs['bb8 offsets'].cpu().numpy() # [B, 16, H, W] # normalized, feature scale (1/4 dino)
    pred_offsets = pred_offsets * STD["bb8_offset"] + MEAN["bb8_offset"]
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

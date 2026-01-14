import numpy as np
import cv2
from pathlib import Path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MEAN = {"center": 0.518, "bb8_offset": 0.0, "center_depth": 6.511, "bb8_depth_offset": -0.007}
STD = {"center": 0.159, "bb8_offset": 0.084, "center_depth": 0.968, "bb8_depth_offset": 0.120}
# Prev_stat
# MEAN = {"center": 0.497, "bb8_offset": 0.0, "center_depth": 6.181, "bb8_depth_offset": -0.009}
# STD = {"center": 0.144, "bb8_offset": 0.097, "center_depth": 1.002, "bb8_depth_offset": 0.133}

def visualization(small_batch, outputs, out_dir: Path):
    preds = outputs # (B, 27)
    p_center_raw = preds[:, 0:2]  # [B, 2]
    p_offsets_raw = preds[:, 2:18]  # [B, 16]
    p_depth_raw = preds[:, 18:19]  # [B, 1]
    p_d_offset_raw = preds[:, 19:27]  # [B, 8]
    
    # Unnormalize predictions
    pred_center = p_center_raw * STD["center"] + MEAN["center"]
    pred_offsets = (p_offsets_raw * STD["bb8_offset"] + MEAN["bb8_offset"]).view(-1, 8, 2)  # [B, 8, 2]
    pred_center = pred_center.detach().cpu().numpy()
    pred_offsets = pred_offsets.detach().cpu().numpy()
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
        pred_center_pixel = pred_center[i] * 512
        pred_offsets_pixel = pred_offsets[i] * 512
        pred_corners = pred_center_pixel[None, :] + pred_offsets_pixel  # [8, 2]
        # Draw GT, Predicted corners
        for j in range(8):
            gt_u, gt_v = int(round(corners_3d[j][0])), int(round(corners_3d[j][1]))
            pd_u, pd_v = int(round(pred_corners[j][0])), int(round(pred_corners[j][1]))
            cv2.circle(image, (gt_u, gt_v), 5, (0,255,0), -1) # Green for GT
            cv2.circle(image, (pd_u, pd_v), 5, (0,0,255), -1) # Red for Pred
        cv2.drawMarker(image, tuple(pred_center_pixel.astype(int)), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
        cv2.imwrite(str(out_dir / f"vis_{i:03d}.png"), image)
    print(f"Visualization images saved to: {out_dir}")

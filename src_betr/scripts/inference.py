import sys
import torch
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
SRC_BETR_DIR = Path(__file__).resolve().parents[1]
PROJ_ROOT = SRC_BETR_DIR.parent
sys.path.insert(0, str(SRC_BETR_DIR))

from models.betr import BETRModel
from data.image_dataloader import build_image_dataloader, _default_transform, filtered_annotations

def inverse_depth_norm(norm_inv_depth, min_depth=0.1, max_depth=3000.0):
    inv_min = 1.0 / max_depth
    inv_max = 1.0 / min_depth
    inv_depth = norm_inv_depth * (inv_max - inv_min) + inv_min
    depth = 1.0 / (inv_depth + 1e-8)
    return depth

def decode_output(output, input_size=512, min_depth=0.1, max_depth=3000.0):
    pred_center_uv = output['center coords'].squeeze(0).squeeze(0) # [2]
    heatmap_x = int(torch.clamp(pred_center_uv[0] / 4, 0, 127).item())
    heatmap_y = int(torch.clamp(pred_center_uv[1] / 4, 0, 127).item())
    
    pred_center_norm = pred_center_uv / input_size
    
    # 3D offsets Sampling
    pred_offsets_3d = output['bb8 offsets']
    pred_offsets = pred_offsets_3d[0, :, heatmap_y, heatmap_x]
    pred_offsets = pred_offsets.view(8, 2)
    
    # Reconsturct 3D Corenrs
    pred_corners_norm = pred_center_norm.unsqueeze(0) + pred_offsets
    corners_pixel = pred_corners_norm * input_size
    
    # Center Depth
    pred_depth_map = output['center depth']
    pred_center_depth_norm = pred_depth_map[0, 0, heatmap_y, heatmap_x]
    pred_center_metric = inverse_depth_norm(pred_center_depth_norm, min_depth, max_depth) 
    
    # Depth Offsets
    pred_depth_offsets_map = output['bb8 depth offsets']
    pred_depth_offsets = pred_depth_offsets_map[0, :, heatmap_y, heatmap_x] # [8]
    pred_corner_depths = pred_center_metric * torch.exp(pred_depth_offsets)
    
    return {
        "center_uv": pred_center_uv.detach().cpu(),
        "corners_pixel": corners_pixel.detach().cpu(),
        "center_depth": pred_center_metric.detach().cpu(),
        "corner_depths": pred_corner_depths.detach().cpu()
    }
            
def print_comparison(gt_data, pred_data, input_size=512):
    print("\n" + "="*80)
    print(f"👀 Result Comparison (Input Size: {input_size}x{input_size})")
    print("="*80)

    # 1. Center Comparison
    gt_c = gt_data['gt_center'] * input_size
    pred_c = pred_data['center_uv']
    diff_c = torch.norm(gt_c - pred_c)
    
    print(f"[📍 Center Position (Pixel)]")
    print(f"  GT   : (x={gt_c[0]:.1f}, y={gt_c[1]:.1f})")
    print(f"  Pred : (x={pred_c[0]:.1f}, y={pred_c[1]:.1f})")
    print(f"  Diff : {diff_c:.2f} px")
    print("-"*80)
    # 2. Metric Depth Comparison
    gt_d = gt_data['gt_metric_depth']
    pred_d = pred_data['center_depth']
    diff_d = abs(gt_d - pred_d)
    
    print(f"[📏 Center Metric Depth]")
    print(f"  GT   : {gt_d:.4f} m")
    print(f"  Pred : {pred_d:.4f} m")
    print(f"  Diff : {diff_d:.4f} m")
    print("-"*80)
    # 3. 3D Bounding Box Corners (UV)
    print(f"[📦 3D Corners (UV Normalized 0~1) & Depth]")
    print(f"{'Idx':<4} | {'GT (u, v)':<20} | {'Pred (u, v)':<20} | {'GT Depth':<10} | {'Pred Depth':<10}")
    print("-"*80)
    
    gt_corners = gt_data['gt_corners_3d'] * 512 # [8, 2]
    pred_corners = pred_data['corners_pixel'] # [8, 2] 
    
    # GT Depth restore (Offsets -> Real Value)
    # gt_depth_offsets = log(depth) - log(center)
    gt_depth_offsets = gt_data['gt_depth_offsets']
    gt_corner_depths = gt_data['gt_metric_depth'] * torch.exp(gt_depth_offsets)
    pred_corner_depths = pred_data['corner_depths']

    for i in range(8):
        gt_u, gt_v = gt_corners[i]
        pd_u, pd_v = pred_corners[i]
        gt_z = gt_corner_depths[i]
        pd_z = pred_corner_depths[i]
        
        print(f"#{i:<3} | ({gt_u:.3f}, {gt_v:.3f})      | ({pd_u:.3f}, {pd_v:.3f})      | {gt_z:.2f} m     | {pd_z:.2f} m")
    print("="*80 + "\n")

def main():
    config = Path(SRC_BETR_DIR / "configs" / "betr_config.yaml")
    checkpoint = Path(SRC_BETR_DIR / "checkpoints" / "betr_model_v1" / "betr_model_v1_best.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(config)
    cfg.feature_mode = False
    
    print(f"Loading model from checkpoint: {checkpoint}")
    model = BETRModel(cfg).to(device)
    checkpoint = torch.load(checkpoint, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}    
    model.load_state_dict(state_dict)
    model.eval()
    
    dataset_root = Path("/home/vmg/Desktop/layout2video/datasets/L2V_new")
    data_dir = Path("/home/vmg/Desktop/layout2video/datasets/")
    dataloader = build_image_dataloader(
        root_dir=dataset_root,
        data_dir=data_dir,
        split="test",
        batch_size=1,
        da3_image_size=448,
        dino_image_size=512,
        shuffle=False,
        num_workers=0,
    )
    dataset = dataloader.dataset
    sample = dataset[100]
    img_da3 = sample['image_da3'].unsqueeze(0).to(device)
    img_dino = sample['image_dino'].unsqueeze(0).to(device)
    bbx_2d = sample['2d_bbx'].unsqueeze(0).to(device)
    mask = sample['padding_mask'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(
            images_da3=img_da3,
            images_dino=img_dino,
            bbx2d_tight=bbx_2d,
            mask=mask,
            feature_mode=False,
        )
        
    pred_decoded = decode_output(output, input_size=512)
    gt_data = {
        "gt_center": sample['gt_center'],
        "gt_corners_3d": sample['gt_corners_3d'],
        "gt_metric_depth": sample['gt_metric_depth'],
        "gt_depth_offsets": sample['gt_depth_offsets'],
    }
    print_comparison(gt_data, pred_decoded, input_size=512)
    
if __name__ == "__main__":
    main()
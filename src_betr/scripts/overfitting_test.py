import torch
import torch.optim as optim
from pathlib import Path
from omegaconf import OmegaConf
import time
import sys
import cv2
import numpy as np

BETR_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = BETR_ROOT.parent
sys.path.insert(0, str(BETR_ROOT))
from models.betr import BETRModel
from losses.criterion import BETRLoss
from data.image_dataloader import build_image_dataloader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def run_overfitting_test():
    print("="*50)
    print("Running Overfitting Test on a Small Dataset")
    print("="*50)
    
    config_path = BETR_ROOT / "configs" / "betr_config.yaml"
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    cfg.batch_size = 4
    model = BETRModel(cfg).to(device)
    gt_size = cfg.data.dino_image_size
    criterion = BETRLoss(cfg.loss_weights.lambda_, cfg.loss_weights.sigma_,
                         heatmap_size=gt_size/4, input_size=gt_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    root_dir = Path(cfg.json_root)
    data_dir = Path(cfg.data_root)
    
    dataloader = build_image_dataloader(
        root_dir=root_dir, data_dir=data_dir, batch_size=cfg.batch_size,
    )
    
    print("Extracting a batch for overfitting...")
    try:
        small_batch = next(iter(dataloader))
    except StopIteration:
        print("DataLoader is empty. Please check the dataset path and contents.")
        return
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in small_batch.items()}
    print("Starting training loop to overfit on the small batch...")
    
    model.train()
    start_time = time.time()
    
    for epoch in range(1, 101):
        optimizer.zero_grad()
        outputs = model(
            images_dino = batch_gpu["image_dino"],
            images_da3 = batch_gpu["image_da3"],
            bbx2d_tight = batch_gpu["2d_bbx"],
            mask = batch_gpu["padding_mask"],
            feature_mode = False
        )
        
        loss_dict = criterion(outputs, batch_gpu)
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        optimizer.step()
        if epoch % 10 ==0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:3d}/300] | Loss: {total_loss.item():.6f} | "
                  f"Center: {loss_dict['loss_center'].item():.6f} | "
                  f"Depth: {loss_dict['loss_depth'].item():.6f} | "
                  f"Offset: {loss_dict['loss_offset'].item():.6f} | "
                  f"Depth Off.: {loss_dict['loss_depth_offset'].item():.6f} | "
                  f"Time: {elapsed:.2f}s")
    print("\n" + "="*50)
    if total_loss.item() < 0.01:
        print("Overfitting test passed! The model successfully overfitted the small batch.")
    else:
        print("Overfitting test failed. The model did not sufficiently reduce the loss.")
    print("="*50)
    model.eval()
    with torch.inference_mode():
        outputs = model(
            bbx2d_tight = batch_gpu["2d_bbx"],
            mask = batch_gpu["padding_mask"],
            f_metric = batch_gpu["feat_metric"],
            f_mono = batch_gpu["feat_mono"],
            f_dino = batch_gpu["feat_dino"],
            feature_mode = True
        )
    return small_batch, outputs
    
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
    pred_center = outputs['center coords'].squeeze(1).cpu().numpy()
    pred_center_depth = outputs['center depth'].cpu().numpy() # [B, 1, H, W]
    pred_depth_offsets = outputs['bb8 depth offsets'].cpu().numpy() # [B, 8, H, W]
    gt_center_depth = small_batch['gt_metric_depth'].cpu() # [B], scalar value
    gt_depth_offsets = small_batch['gt_offsets_3d'].cpu().numpy() # [B, 8]
    for i in range(4):
        gt_depths = gt_center_depth[i].item() * np.exp(gt_depth_offsets[i])  # [8]
        heatmap_center_x, heatmap_center_y = int(np.clip(round(pred_center[i][0]/4), 0, 127)), int(np.clip(round(pred_center[i][1]/4), 0, 127))
        pred_offset = pred_depth_offsets[i,:,heatmap_center_y, heatmap_center_x] # [8]
        pred_center = pred_center_depth[i,0,heatmap_center_y, heatmap_center_x].item()
        pred_depths = pred_center * np.exp(pred_offset)  # [8]
        print(f"Sample #{i}:")
        print(f"{'Idx':<4} | {'GT Depth (m)':<15} | {'Pred Depth (m)':<15}")
        print("-"*40)
        for j in range(8):
            gt_z = gt_depths[j].item()
            pd_z = pred_depths[i][j].item()
            print(f"#{j:<3} | {gt_z:<15.2f} | {pd_z:<15.2f}")
        print("="*40 + "\n")
        

if __name__ == "__main__":
    small_batches, outputs = run_overfitting_test()
    output_dir = PROJ_ROOT / "test" / "overfitting_vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization(small_batches, outputs, output_dir)
    comparing_depth(small_batches, outputs)
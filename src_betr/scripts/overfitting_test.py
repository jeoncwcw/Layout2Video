import torch
import torch.optim as optim
from pathlib import Path
from omegaconf import OmegaConf
import time
import sys

BETR_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = BETR_ROOT.parent
sys.path.insert(0, str(BETR_ROOT))
from models.betr_v2 import BETRModel2
from models.betr import BETRModel
from losses.criterion_v2 import BETRv2Loss
from losses.criterion import BETRLoss
from data.image_dataloader import build_image_dataloader
from utils import set_seed, visualization, visualize_heatmaps_feature_mode


def run_overfitting_test():
    print("="*50)
    print("Running Overfitting Test on a Small Dataset")
    print("="*50)
    
    config_path = BETR_ROOT / "configs" / "betr_config.yaml"
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.feature_mode = False
    cfg.device = str(device)
    cfg.batch_size = 4
    model = BETRModel(cfg).to(device)
    criterion = BETRLoss(
        lambda_fine=cfg.loss_weights.lambda_,
        sigma=cfg.loss_weights.sigma_,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    root_dir = Path(cfg.json_root)
    data_dir = Path(cfg.data_root)
    set_seed(cfg.get("seed", 42))
    dataloader = build_image_dataloader(
        root_dir=root_dir, data_dir=data_dir, batch_size=cfg.batch_size, shuffle=True, seed=42,
    )
    
    print("Extracting a batch for overfitting...")
    small_batch = next(iter(dataloader))
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in small_batch.items()}
    print("Starting training loop to overfit on the small batch...")
    
    model.train()
    start_time = time.time()
    
    for epoch in range(1, 301):
        optimizer.zero_grad()
        outputs = model(
            images_dino = batch_gpu["image_dino"],
            images_da3 = batch_gpu["image_da3"],
            bbx2d_tight = batch_gpu["2d_bbx"],
            mask = batch_gpu["padding_mask"],
        )
        
        loss_dict = criterion(outputs, batch_gpu)
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        optimizer.step()
        if epoch % 10 ==0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:3d}/300] | Loss: {total_loss.item():.6f} | "
                  f"Corners: {loss_dict['loss_corners'].item():.6f} | "
                  f"Depths: {loss_dict['loss_depths'].item():.6f} | "
                  f"Time: {elapsed:.2f}s")
    print("\n" + "="*50)
    if total_loss.item() < 0.01:
        print("Overfitting test passed! The model successfully overfitted the small batch.")
    else:
        print("Overfitting test failed. The model did not sufficiently reduce the loss.")
    print("="*50)
    model.eval()
    vis_dir = PROJ_ROOT / "test" / "overfitting_vis" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        outputs = model(
            images_dino = batch_gpu["image_dino"],
            images_da3 = batch_gpu["image_da3"],
            bbx2d_tight = batch_gpu["2d_bbx"],
            mask = batch_gpu["padding_mask"],
        )
        idx = 0 
        h_maps = outputs['corner heatmaps'][idx] # [8, 128, 128]
        
        p_coords_128 = outputs['corner coords'][idx] / 4.0 
        
        g_coords_128 = small_batch['gt_corners'][idx] * 128.0
        
        save_name = vis_dir / f"epoch_{epoch+1:03d}_feat_vis.png"
        visualize_heatmaps_feature_mode(h_maps, p_coords_128, g_coords_128, save_name)
        print(f"📸 [Rank 0] Feature-base heatmap saved: {save_name}")
    return small_batch, outputs

        

if __name__ == "__main__":
    small_batches, outputs = run_overfitting_test()
    output_dir = PROJ_ROOT / "test" / "overfitting_vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization(small_batches, outputs, output_dir)
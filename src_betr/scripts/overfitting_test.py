import torch
import torch.optim as optim
from pathlib import Path
from omegaconf import OmegaConf
import time
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from models.betr import BETRModel
from losses.criterion import BETRLoss
from data.wds_dataloader import build_wds_feature_dataloader


def run_overfitting_test():
    print("="*50)
    print("Running Overfitting Test on a Small Dataset")
    print("="*50)
    
    config_path = PROJECT_ROOT / "configs" / "betr_config.yaml"
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    
    model = BETRModel(cfg).to(device)
    gt_size = cfg.data.dino_image_size
    criterion = BETRLoss(cfg.loss_weights.lambda_, cfg.loss_weights.sigma_,
                         heatmap_size=gt_size/4, input_size=gt_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    root_dir = Path(cfg.wds_root)
    
    dataloader = build_wds_feature_dataloader(
        wds_root=root_dir,
        batch_size=cfg.batch_size,
        dino_img_size=gt_size,
        split="train",
        num_workers=4,
        epoch_length=cfg.epoch_length
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
    
    for epoch in range(1, 301):
        optimizer.zero_grad()
        outputs = model(
            bbx2d_tight = batch_gpu["2d_bbx"],
            mask = batch_gpu["padding_mask"],
            f_metric = batch_gpu["feat_metric"],
            f_mono = batch_gpu["feat_mono"],
            f_dino = batch_gpu["feat_dino"],
            feature_mode = True
        )
        
        loss_dict = criterion(outputs, batch_gpu)
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        optimizer.step()
        if epoch % 20 ==0:
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
    
if __name__ == "__main__":
    run_overfitting_test()
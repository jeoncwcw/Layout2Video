import os
import torch
import torch.optim as optim
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import time
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.betr import BETRModel
from losses.criterion import BETRLoss
from data.feature_dataloader import build_feature_dataloader
from data.image_dataloader import build_image_dataloader

def train():
    # 1. Load configuration
    config_path = PROJECT_ROOT / "configs" / "betr_config.yaml"
    cfg = OmegaConf.load(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    print(f"Using device: {device}")
    
    # 2. Initialize model, loss function, and optimizer
    model = BETRModel(cfg).to(device)
    gt_size = cfg.data.dino_image_size
    criterion = BETRLoss(
        cfg.loss_weights.lambda_, 
        cfg.loss_weights.sigma_,
        heatmap_size=gt_size/4, 
        input_size=gt_size
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    
    # 3. Prepare data loaders
    train_data_kwargs = {
        "root_dir": Path(cfg.json_root),
        "feature_dir": Path(cfg.feature_dir),
        "image_map_pth": Path(cfg.image_map_path),
        "batch_size": cfg.batch_size,
        "dino_img_size": gt_size,
        "target_quality": cfg.data.target_quality,
        "min_area": cfg.data.min_area_object,
        "num_workers": cfg.num_workers,
    }
    train_dataloader = build_feature_dataloader(**train_data_kwargs)
    val_data_kwargs = {
        
    }
    
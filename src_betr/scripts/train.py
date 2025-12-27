import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.betr import BETRModel
from losses.criterion import BETRLoss
from data.feature_dataloader import build_feature_dataloader

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def reduce_loss(loss, world_size):
    rt = loss.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

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
    total_data_kwargs = {
        "root_dir": Path(cfg.json_root),
        "feature_dir": Path(cfg.feature_dir),
        "image_map_pth": Path(cfg.image_map_path),
        "batch_size": cfg.batch_size,
        "dino_img_size": gt_size,
        "target_quality": cfg.data.target_quality,
        "min_area": cfg.data.min_area_object,
        "num_workers": cfg.num_workers,
    }
    train_data_kwargs = {**total_data_kwargs, "split": "train"}
    val_data_kwargs = {**total_data_kwargs, "split": "val"}
    train_dataloader = build_feature_dataloader(**train_data_kwargs)
    val_dataloader = build_feature_dataloader(**val_data_kwargs)
    
    num_epochs = cfg.num_epochs
    best_val_loss = float('inf')
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print(f"Starting training for {num_epochs} epochs...")
    print("=" * 50)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        
        for batch in pbar:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(
                bbx2d_tight=batch_gpu["2d_bbx"],
                mask=batch_gpu["padding_mask"],
                feat_metric=batch_gpu["feat_metric"],
                feat_mono=batch_gpu["feat_mono"],
                feat_dino=batch_gpu["feat_dino"],
                feature_mode=cfg.model.feature_mode,
            )
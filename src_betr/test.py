import torch
import torch.optim as optim
from pathlib import Path
from omegaconf import OmegaConf
import time
import sys

BETR_ROOT = Path(__file__).resolve().parents[0]
PROJ_ROOT = BETR_ROOT.parent
sys.path.insert(0, str(BETR_ROOT))
from models.betr import BETRModel
from data.image_dataloader import build_image_dataloader
from utils import set_seed

def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def run_forward_pass_test():
    print("="*50)
    print("Running Model to check forward pass")
    print("="*50)
    
    config_path = BETR_ROOT / "configs" / "betr_config.yaml"
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    cfg.batch_size = 2
    cfg.feature_mode = False  # Use feature mode for faster testing
    model = BETRModel(cfg).to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: total={total_params:,} | trainable={trainable_params:,}")
    
    root_dir = Path(cfg.json_root)
    data_dir = Path(cfg.data_root)
    set_seed(cfg.get("seed", 42))
    dataloader = build_image_dataloader(
        root_dir=root_dir, data_dir=data_dir, batch_size=cfg.batch_size, shuffle=True, seed=42,
    )
    
    print("Extracting a batch for testing...")
    small_batch = next(iter(dataloader))
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in small_batch.items()}
    print("Running model forward pass...")
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            images_dino = batch_gpu["image_dino"],
            images_da3 = batch_gpu["image_da3"],
            bbx2d_tight = batch_gpu["2d_bbx"],
            mask = batch_gpu["padding_mask"],
        )
    return outputs
    


if __name__ == "__main__":
    run_forward_pass_test()
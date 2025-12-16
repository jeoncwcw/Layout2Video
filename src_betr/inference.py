import sys
import torch
import traceback
from omegaconf import OmegaConf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from models.betr import BETRModel
from data.image_dataloader import build_image_dataloader

FILE_PATH = Path(__file__).resolve()
SRC_BETR_DIR = FILE_PATH.parent  # src_betr/
PROJECT_ROOT = SRC_BETR_DIR.parent  # Project Root (Layout2Video-main/)

def resolve_path(cfg, key, root):
    if key in cfg and cfg[key]:
        path = Path(cfg[key])
        if not path.is_absolute():
            cfg[key] = str(root / path)
            
            
def test_inference():
    print("Starting inference test...")
    
    config_path = SRC_BETR_DIR / "configs" / "betr_config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return
    cfg = OmegaConf.load(config_path)
    path_keys = [
        "monodepth_cfg_path", "monodepth_checkpoint_path",
        "metricdepth_cfg_path", "metricdepth_checkpoint_path",
        "dinov3_checkpoint_path"
    ]
    for key in path_keys:
        resolve_path(cfg, key, PROJECT_ROOT)   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    model = BETRModel(cfg).to(device)
    
    dataloader = build_image_dataloader(
        root_dir=Path("/home/vmg/Desktop/layout2video/datasets/L2V/labeled"),
        data_dir=Path("/home/vmg/Desktop/layout2video/datasets"),
        split="test",
        batch_size=2,
        num_workers=2,
        shuffle=False,
    )
    
    for batch in dataloader:
        batch_da3 = batch["image_da3"].to(device)
        batch_dino = batch["image_dino"].to(device)
        bbx2d_tight = batch["2d_bbx"].to(device)
        mask = batch["padding_mask"].to(device)
        breakpoint()
        
        model.eval()
        with torch.no_grad():
            output = model(batch_da3, batch_dino, bbx2d_tight, mask)
            breakpoint()
            break
            
if __name__ == "__main__":
    test_inference()
    
    

import torch
from omegaconf import OmegaConf
from pathlib import Path

import sys
SRC_BETR_DIR = Path(__file__).resolve().parents[1]
PROJ_ROOT = SRC_BETR_DIR.parent
sys.path.insert(0, str(SRC_BETR_DIR))

from models.betr import BETRModel
from data.image_dataloader import build_image_dataloader
from utils import set_seed, visualization

def main():
    config = Path(SRC_BETR_DIR / "configs" / "betr_config.yaml")
    checkpoint = Path(SRC_BETR_DIR / "checkpoints" / "betr_model_corners_v1" / "betr_model_corners_v1_epoch100.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(config)
    cfg.feature_mode = False
    cfg.batch_size = 16
    set_seed(cfg.get("seed", 40))
    
    print(f"Loading model from checkpoint: {checkpoint}")
    model = BETRModel(cfg).to(device)
    checkpoint = torch.load(checkpoint, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}    
    model.load_state_dict(state_dict)
    model.eval()
    
    dataset_root = Path(cfg.json_root)
    data_dir = Path(cfg.data_root)
    
    dataloader = build_image_dataloader(root_dir=dataset_root, data_dir=data_dir, shuffle=True, seed=40, batch_size=cfg.batch_size)
    small_batch = next(iter(dataloader))
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in small_batch.items()}
    print(f"image_paths: {batch_gpu['path']}")
    # Inference
    with torch.inference_mode():
        output = model(
            images_da3=batch_gpu["image_da3"],
            images_dino=batch_gpu["image_dino"],
            bbx2d_tight=batch_gpu["2d_bbx"],
            mask=batch_gpu["padding_mask"],
        )
    # Visualization & Depth Stats
    output_dir = PROJ_ROOT / "test" / "inference_vis_corners_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization(small_batch, output, output_dir)
    
    
if __name__ == "__main__":
    main()
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_betr.models.feature_extractor import (
    da3_predict, dinov3_predict
)
'''
images: Sequence[str | np.ndarray | Image.Image],
    *,
    cfg_path: Path = DA3_METRIC_CFG,
    checkpoint_path: Path = DA3_METRIC_CKPT,
    device: str | torch.device | None = "auto",
    is_metric: bool | None = None,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
) -> torch.Tensor:
'''

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    
    img_path = "/home/vmg/Desktop/layout2video/datasets/KITTI_object/testing/image_2/000000.png"
    image = Image.open(img_path).convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_root = str(Path(__file__).resolve().parents[0] / "checkpoints")
    cfg_root = str(Path(__file__).resolve().parents[0] / "configs")
    da3_feat = da3_predict([image],
                           cfg_path=Path(cfg_root + "/da3metric-large.yaml"),
                           checkpoint_path=f"{checkpoint_root}/DA3Metric_Large.safetensors",
                           device = device,
                           )
    dinov3_feat = dinov3_predict([image],
                                 checkpoint_path=f"{checkpoint_root}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                                 device = device,
            )
    print("DA3 feature shape:", da3_feat.shape)
    print("DINOv3 feature shape:", dinov3_feat.shape)
    

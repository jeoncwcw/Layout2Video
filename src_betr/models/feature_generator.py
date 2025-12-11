import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src_betr.models.feature_extractor import (
    da3_predict, dinov3_predict
)

class FeatureGenerator(nn.Module):
    def __init__(self):
        super(FeatureGenerator, self).__init__()
        # Initialize layers or parameters here
        self.patchify_conv = nn.Conv2d(
            in_channels=1,
            out_channels=512,
            kernel_size=14,
            stride=14,
        )
    def forward(self, o_metric, o_mono, o_dino3):
        patchified_metric = self.patchify_conv(o_metric)
        patchified_mono = self.patchify_conv(o_mono)
        da3_features = torch.concat([patchified_metric, patchified_mono], dim=1)
        combined_features = da3_features + o_dino3
        return combined_features
    
if __name__ == "__main__":
    from PIL import Image
    
    img_path = "/home/vmg/Desktop/layout2video/datasets/KITTI_object/testing/image_2/000000.png"
    image = Image.open(img_path).convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_root = str(Path(__file__).resolve().parents[1] / "checkpoints")
    cfg_root = str(Path(__file__).resolve().parents[1] / "configs")
    da3_metric_feat = da3_predict([image],
                           cfg_path=Path(cfg_root + "/da3metric-large.yaml"),
                           checkpoint_path=f"{checkpoint_root}/DA3Metric_Large.safetensors",
                           device = device,
                           )
    da3_mono_feat = da3_predict([image],
                           cfg_path=Path(cfg_root + "/da3mono-large.yaml"),
                           checkpoint_path=f"{checkpoint_root}/DA3Mono_Large.safetensors",
                           device = device,
                           )
    dinov3_feat = dinov3_predict([image],
                                 checkpoint_path=f"{checkpoint_root}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                                 device = device,
            )
    feature_generator = FeatureGenerator()
    feature_generator.to(device)
    feature_generator.eval()
    with torch.no_grad():
        output_features = feature_generator(da3_metric_feat, da3_mono_feat, dinov3_feat)
    print("Output feature shape:", output_features.shape)
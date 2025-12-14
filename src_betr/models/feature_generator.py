import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src_betr.models.feature_extractor import (
    da3_predict, dinov3_predict
)
from src_betr.data.image_dataloader import build_image_dataloader

class FeatureGenerator(nn.Module):
    def __init__(self):
        super(FeatureGenerator, self).__init__()
        # Initialize layers or parameters here
        self.patchify_conv_mono = nn.Conv2d(
            in_channels=1,
            out_channels=512,
            kernel_size=14,
            stride=14,
        )
        self.patchify_conv_metric = nn.Conv2d(
            in_channels=1,
            out_channels=512,
            kernel_size=14,
            stride=14,
        )
        self.norm_metric = nn.InstanceNorm2d(1, affine=True)
        self.norm_mono = nn.InstanceNorm2d(1, affine=True)
    def forward(self, o_metric, o_mono, o_dino3):
        patchified_metric = self.patchify_conv_metric(self.norm_metric(torch.log1p(o_metric)))
        patchified_mono = self.patchify_conv_mono(self.norm_mono(o_mono))
        da3_features = torch.concat([patchified_metric, patchified_mono], dim=1)
        combined_features = torch.concat([da3_features, o_dino3], dim=1)
        return combined_features
    
def running_feature_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = str(Path(__file__).resolve().parents[2] / "datasets" / "KITTI_object" / "testing" / "image_2")
    dataloader = build_image_dataloader(
        data_dir=dataset_root,
        batch_size=2,
        da3_image_size=448,
        dino_image_size=512,
        num_workers=0,
    )
    checkpoint_root = str(Path(__file__).resolve().parents[1] / "checkpoints")
    cfg_root = str(Path(__file__).resolve().parents[1] / "configs")
    for batch in dataloader:
        batch_da3 = batch["image_da3"].to(device)
        batch_dino = batch["image_dino"].to(device)
            
        o_metric = da3_predict(
            images = batch_da3,
            cfg_path=Path(cfg_root + "/da3metric-large.yaml"),
            checkpoint_path=f"{checkpoint_root}/DA3Metric_Large.safetensors",
            device = device,
        )
        o_mono = da3_predict(
            images = batch_da3,
            cfg_path=Path(cfg_root + "/da3mono-large.yaml"),
            checkpoint_path=f"{checkpoint_root}/DA3Mono_Large.safetensors",
            device = device,
        )
        o_dino3 = dinov3_predict(
            images = batch_dino,
            checkpoint_path=f"{checkpoint_root}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            device = device,
        )
        feature_generator = FeatureGenerator()
        feature_generator.to(device)
        feature_generator.eval()
        with torch.no_grad():
            output_features = feature_generator(o_metric, o_mono, o_dino3)
        print("Output feature shape:", output_features.shape)
        break
    
if __name__ == "__main__":
    running_feature_generator()
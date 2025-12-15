import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from models.feature_generator import FeatureGenerator
from data.image_dataloader import build_image_dataloader
from models.feature_extractor import DA3FeatureExtractor, DINOv3FeatureExtractor
import torch

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    checkpoint_dir, cfg_dir = str(ROOT / "checkpoints"), str(ROOT / "configs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = build_image_dataloader(
        root_dir=Path(""),
        data_dir=Path(""),
        split="test",
        batch_size=2,
        num_workers=2,
        shuffle=False,
    )
    metric_depth_extractor = DA3FeatureExtractor(
        cfg_path=Path(cfg_dir + "/da3metric-large.yaml"), checkpoint_path=f"{checkpoint_dir}/DA3Metric_Large.safetensors", device=device
    )
    mono_depth_extractor = DA3FeatureExtractor(
        cfg_path=Path(cfg_dir + "/da3mono-large.yaml"), checkpoint_path=f"{checkpoint_dir}/DA3Mono_Large.safetensors", device=device,
    )
    dinov3_extractor = DINOv3FeatureExtractor(
        checkpoint_path=f"{checkpoint_dir}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", device=device,
    )
    feature_generator = FeatureGenerator().to(device)

    for batch in dataloader:
        batch_da3 = batch["image_da3"].to(device)
        batch_dino = batch["image_dino"].to(device)

        o_metric = metric_depth_extractor(batch_da3)
        o_mono = mono_depth_extractor(batch_da3)
        o_dino3 = dinov3_extractor(batch_dino)
        breakpoint()

        feature_generator.eval()
        with torch.no_grad():
            output_features = feature_generator(o_metric, o_mono, o_dino3)
        breakpoint()
        break
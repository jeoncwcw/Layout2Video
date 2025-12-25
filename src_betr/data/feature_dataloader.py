import torch
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.utils import balanced_sampler, filtered_annotations

class FeatureDataset(Dataset):
    def __init__(self, json_data: list, feature_dir: str | Path, image_map_pth: str | Path, dino_img_size: int=512) -> None:
        self.feature_dir = Path(feature_dir)
        self.dino_img_size = dino_img_size

        with open(image_map_pth, 'r') as f:
            self.image_map = json.load(f)

        self.ann_list = []
        self.image_info_map = {}

        for data in json_data:
            for img_info in data["images"]:
                self.image_info_map[img_info["id"]] = img_info
            for obj in data["annotations"]:
                img_id = obj["image_id"]
                img_info = self.image_info_map[img_id]

                if img_info:
                    rel_path = img_info["file_path"]
                    if rel_path in self.image_map:
                        self.ann_list.append({
                            "image_id": obj["image_id"],
                            "2d_bbx": obj["bbox2D_tight"],
                            "3d_bb8": self._convert_projected_corners(obj["projected_corners"]),
                            "quality": obj["quality"],
                            "depth": obj["depth"],
                            "rel_path": rel_path
                        })
                    else:
                        print(f"Warning: Image path {rel_path} not found in image map.")

    def _convert_projected_corners(self, corners_list: List[Dict]):
        coords = [(float(c["u"]), float(c["v"])) for c in corners_list]
        return torch.tensor(coords, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ann_list)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        # Processing input images
        ann_data = self.ann_list[index]
        img_id = ann_data["image_id"]
        img_info = self.image_info_map[img_id]
        rel_path = ann_data["rel_path"]

        feat_file = self.image_map[rel_path]
        feat_path = self.feature_dir / feat_file

        features = torch.load(feat_path, map_location="cpu")

        # Applying Letterboxing transformations to bounding boxes
        orig_w, orig_h = img_info["width"], img_info["height"]
        longest = max(orig_w, orig_h)
        scale = self.dino_img_size / float(longest)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        
        pad_left, pad_top = (self.dino_img_size - new_w) // 2, (self.dino_img_size - new_h) // 2
        bbx2d_orig = np.array(ann_data["2d_bbx"], dtype=np.float32)
        bbx2d_processed = bbx2d_orig * scale
        bbx2d_processed[[0, 2]] += pad_left
        bbx2d_processed[[1, 3]] += pad_top
        bbx2d_processed = torch.tensor(bbx2d_processed, dtype=torch.float32) / self.dino_img_size

        # 3D bb8 processing
        bbx3d_bb8 = ann_data["3d_bb8"] * scale
        bbx3d_bb8[:, 0] += pad_left
        bbx3d_bb8[:, 1] += pad_top
        bbx3d_bb8 = bbx3d_bb8 / self.dino_img_size # Normalize to [0,1] range
        bbx3d_center = bbx3d_bb8.mean(dim=0) # [2]
        offsets_3d = bbx3d_bb8 - bbx3d_center.unsqueeze(0) # Normalized offsets

        # depth (Metric to Stable Monodepth Style)
        raw_depths = torch.tensor(ann_data["depth"], dtype=torch.float32)
        center_metric_depth = raw_depths.mean()

        min_depth, max_depth = 0.1, 3000.0
        center_depth_clamped = torch.clamp(center_metric_depth, min=min_depth, max=max_depth)
        inv_depth = 1.0 / center_depth_clamped
        inv_min = 1.0 / max_depth
        inv_max = 1.0 / min_depth
        norm_inv_depth = (inv_depth - inv_min) / (inv_max - inv_min) # Normalized inverse depth [0,1]

        raw_depth_clamped = torch.clamp(raw_depths, min=min_depth, max=max_depth)
        depth_offsets = torch.log(raw_depth_clamped) - torch.log(center_depth_clamped) # Log space offsets (not need to normalize)
        
        # Generating key padding_mask for transformer
        padding_mask = torch.ones((self.dino_img_size, self.dino_img_size), dtype=torch.bool)
        padding_mask[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = False # [H, W] (True for padding)

        return {
            # Feature Tensors of Backbone Outputs (Removing batch dimension)
            "feat_metric": features["metric"].float().squeeze(0),
            "feat_mono": features["mono"].float().squeeze(0),
            "feat_dino": features["dino"].float().squeeze(0),

            # Basic infos
            "path": str(rel_path), "2d_bbx": bbx2d_processed, "quality": ann_data["quality"],
            ## GT infos
            "gt_center": bbx3d_center, "gt_offsets_3d": offsets_3d.flatten(), "gt_corners_3d": bbx3d_bb8,
            "gt_center_depth": norm_inv_depth, "gt_depth_offsets": depth_offsets,
            "meta_depth_min": min_depth, "meta_depth_max": max_depth, "gt_metric_depth": center_metric_depth,
            # padding mask
            "padding_mask": padding_mask,
            }
    
def build_feature_dataloader(
    root_dir: Path,
    feature_dir: Path,
    image_map_pth: Path,
    batch_size: int = 16,
    dino_img_size: int = 512,
    target_quality: str = "Good",
    min_area: int = 32*32,
    num_workers: int = 4,
) -> DataLoader:
    json_paths = sorted(root_dir.glob(f"*train.json"))

    json_list = []
    for p in json_paths:
        data = filtered_annotations(
            p,
            target_quality=target_quality,
            min_area=min_area,
            dino_size=dino_img_size
        )
        json_list.append(data)

    dataset = FeatureDataset(
        json_data=json_list,
        feature_dir=feature_dir,
        image_map_pth=image_map_pth,
        dino_img_size=dino_img_size
    )
    sampler = balanced_sampler(json_paths, json_list)
    shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )

if __name__ == "__main__":
    # Example: iterate over a folder full of images
    sample_root = Path("/home/vmg/Desktop/layout2video/datasets/L2V_new")
    feature_dir = Path("/home/vmg/Desktop/layout2video/datasets/betr_features")
    loader = build_feature_dataloader(sample_root, feature_dir, image_map_pth = Path("/home/vmg/Desktop/layout2video/datasets/image_map.json"), batch_size=2, dino_img_size=512, num_workers=0)
    batch = next(iter(loader))
    print("Batch feat metric depth tensor shape:", batch["feat_metric"].shape)
    print("Batch feat mono depth tensor shape:", batch["feat_mono"].shape)
    print("Batch feat dino tensor shape:", batch["feat_dino"].shape)
    print("Batch file paths:", batch["path"])
    print("length of dataset:",len(loader.dataset))
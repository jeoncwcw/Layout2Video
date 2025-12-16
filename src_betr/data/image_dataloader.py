from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def _LetterBoxing(img: Image.Image, target_size: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest == target_size:
        return img
    scale = target_size / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
    delta_w = target_size - new_w
    delta_h = target_size - new_h
        
    pad_left = delta_w // 2; pad_right = delta_w - pad_left
    pad_top = delta_h // 2; pad_bottom = delta_h - pad_top 
    padded_arr = cv2.copyMakeBorder(
        arr,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0,0,0)
    )
    return Image.fromarray(padded_arr)

def _default_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    """
    Resize to a square (if requested), convert to CHW float tensor, and normalize.
    """

    def transform(img: Image.Image) -> torch.Tensor:
        if image_size and image_size > 0:
            img = _LetterBoxing(img, image_size)
        arr = torch.from_numpy(np.array(img, copy=True)).float() / 255.0
        arr = arr.permute(2, 0, 1)  # HWC -> CHW
        mean = arr.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = arr.new_tensor(IMAGENET_STD).view(3, 1, 1)
        return (arr - mean) / std

    return transform


class AnnotationDataset(Dataset):
    """
    Minimal dataset that collects every image file under a root directory.
    """

    def __init__(
        self,
        json_data: List[Dict],
        root_dir: str | Path,
        transform_da3: Callable[[Image.Image], torch.Tensor],
        transform_dino: Callable[[Image.Image], torch.Tensor],
        dino_image_size: int = 512,
    ) -> None:
        self.root = Path(root_dir).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        self.transform_da3 = transform_da3
        self.transform_dino = transform_dino
        self.dino_image_size = dino_image_size

        self.ann_list = []
        self.image_map = {}

        for data in json_data:
            for img_info in data["images"]:
                img_path = self.root / img_info["file_path"]
                self.image_map[img_info["id"]] = {"path": img_path, "info": img_info}
            for obj in data["annotations"]:
                self.ann_list.append({
                    "image_id": obj["image_id"],
                    "2d_bbx": obj["bbox2D_tight"],
                    "3d_bb8": self._convert_projected_corners(obj["projected_corners"]),
                    "quality": obj["quality"],
                    "depth": obj["depth"],
                })

    def _convert_projected_corners(self, corners_list: List[Dict]):
        coords = [(float(c["u"]), float(c["v"])) for c in corners_list]
        return torch.tensor(coords, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ann_list)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        # Processing input images
        ann_data = self.ann_list[index]
        img_id = ann_data["image_id"]
        img_info = self.image_map[img_id]
        img_path = img_info["path"]
        image = Image.open(img_path).convert("RGB")

        image_da3 = self.transform_da3(image)
        image_dino = self.transform_dino(image)

        # Applying Letterboxing transformations to bounding boxes
        bbx2d_orig = np.array(ann_data["2d_bbx"], dtype=np.float32)
        longest = max(image.width, image.height)
        scale = self.dino_image_size / float(longest) 
        new_w, new_h = int(round(image.width * scale)), int(round(image.height * scale))
        pad_left, pad_top = (self.dino_image_size - new_w) // 2, (self.dino_image_size - new_h) // 2
        bbx2d_processed = bbx2d_orig * scale
        bbx2d_processed[[0, 2]] += pad_left
        bbx2d_processed[[1, 3]] += pad_top
        bbx2d_processed = torch.tensor(bbx2d_processed, dtype=torch.float32) / self.dino_image_size
        bbx3d_bb8 = ann_data["3d_bb8"] * scale
        bbx3d_bb8[:, 0] += pad_left
        bbx3d_bb8[:, 1] += pad_top
        bbx3d_bb8 = bbx3d_bb8 / self.dino_image_size
        bbx3d_center = bbx3d_bb8.mean(dim=0)
        offsets_3d = bbx3d_bb8 - bbx3d_center.unsqueeze(0)

        # depth (Metric to Stable Monodepth Style)
        raw_depths = torch.tensor(ann_data["depth"], dtype=torch.float32)
        center_metric_depth = raw_depths.mean()

        min_depth, max_depth = 0.1, 3000.0
        center_depth_clamped = torch.clamp(center_metric_depth, min=min_depth, max=max_depth)
        inv_depth = 1.0 / center_depth_clamped
        inv_min = 1.0 / max_depth
        inv_max = 1.0 / min_depth
        norm_inv_depth = (inv_depth - inv_min) / (inv_max - inv_min)

        raw_depth_clamped = torch.clamp(raw_depths, min=min_depth, max=max_depth)
        depth_offsets = torch.log(raw_depth_clamped) - torch.log(center_depth_clamped)
        
        # Generating key padding_mask for transformer
        padding_mask = torch.ones((self.dino_image_size, self.dino_image_size), dtype=torch.bool)
        padding_mask[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = False

        return {
            "image_da3": image_da3, "image_dino": image_dino, "path": str(img_path), 
            "2d_bbx": bbx2d_processed, "quality": ann_data["quality"],
            ## GT infos
            "gt_center": bbx3d_center, "gt_offsets_3d": offsets_3d.flatten(), "gt_corners_3d": bbx3d_bb8,
            "gt_center_depth": norm_inv_depth, "gt_depth_offsets": depth_offsets,
            "meta_depth_min": min_depth, "meta_depth_max": max_depth, "gt_metric_depth": center_metric_depth,
            # padding mask
            "padding_mask": padding_mask,
            }
    


def build_image_dataloader(
    root_dir: Path,
    data_dir: Path,
    split: str = "test", 
    batch_size: int = 8,
    da3_image_size: int = 448,
    dino_image_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader from a directory that only contains images.
    """
    json_paths = sorted(root_dir.glob(f"*{split}.json"))
    json_list = [json.loads(p.read_text(encoding="utf-8")) for p in json_paths]
    dataset = AnnotationDataset(
        json_data=json_list,
        root_dir=data_dir,
        transform_da3=_default_transform(da3_image_size),
        transform_dino=_default_transform(dino_image_size),
        dino_image_size=dino_image_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


if __name__ == "__main__":
    # Example: iterate over a folder full of images
    sample_root = Path("/home/vmg/Desktop/layout2video/datasets/L2V/labeled")
    data_root = Path("/home/vmg/Desktop/layout2video/datasets")
    loader = build_image_dataloader(sample_root, data_root, split="val",batch_size=2, da3_image_size=448, dino_image_size=512, num_workers=0)
    batch = next(iter(loader))
    print("Batch DA3 image tensor shape:", batch["image_da3"].shape)
    print("Batch DINO image tensor shape:", batch["image_dino"].shape)
    print("Batch file paths:", batch["path"])
    print("min/max DA3 image pixel values:", batch["image_da3"].min().item(), batch["image_da3"].max().item())
    print("min/max DINO image pixel values:", batch["image_dino"].min().item(), batch["image_dino"].max().item())
    print("length of dataset:",len(loader.dataset))
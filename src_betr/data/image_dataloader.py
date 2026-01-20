from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import json
import random
import functools

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.utils import balanced_sampler, filtered_annotations

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# # TODO: removing hard-coded mean and std values
# MEAN = {"center": 0.518, "bb8_offset": 0.0, "center_depth": 6.511, "bb8_depth_offset": -0.007}
# STD = {"center": 0.159, "bb8_offset": 0.084, "center_depth": 0.968, "bb8_depth_offset": 0.120}
# Prev_stat
# MEAN = {"center": 0.497, "bb8_offset": 0.0, "center_depth": 6.181, "bb8_depth_offset": -0.009}
# STD = {"center": 0.144, "bb8_offset": 0.097, "center_depth": 1.002, "bb8_depth_offset": 0.133}

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

def _seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
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
                    "3d_bb8": obj["projected_corners"],
                    "depth": obj["depth"],
                })

    def __len__(self) -> int:
        return len(self.ann_list)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        # Processing input images
        ann_data = self.ann_list[index]
        img_info = self.image_map[ann_data["image_id"]]
        image = Image.open(img_info["path"]).convert("RGB")

        image_da3 = self.transform_da3(image)
        image_dino = self.transform_dino(image)

        # Applying Letterboxing transformations to bounding boxes
        w, h = image.size
        longest = max(w, h)
        scale = self.dino_image_size / float(longest)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        pad_left = (self.dino_image_size - new_w) // 2
        pad_top = (self.dino_image_size - new_h) // 2
        
        raw_corners = torch.tensor([(float(c["u"]), float(c["v"])) for c in ann_data["3d_bb8"]], dtype=torch.float32)
        gt_corners = raw_corners * scale
        gt_corners[:, 0] += pad_left
        gt_corners[:, 1] += pad_top
        gt_corners = gt_corners / self.dino_image_size  
        #### Normalize to [0,1] ####
        
        # 2D Bounding Box
        bbx2d = torch.tensor(ann_data["2d_bbx"], dtype=torch.float32) * scale
        bbx2d[[0,2]] += pad_left
        bbx2d[[1,3]] += pad_top
        bbx2d = bbx2d / self.dino_image_size
        #### Normalize to [0,1] ####
        
        # depths
        raw_depths = torch.clamp(torch.tensor(ann_data["depth"], dtype=torch.float32), min=1e-3)
        
        gt_depth_corners = torch.log(raw_depths + 1e-8)
        padding_mask = torch.ones((self.dino_image_size, self.dino_image_size), dtype=torch.bool)
        padding_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = False
        
        
        return {
            # Input Values
            "image_da3": image_da3, "image_dino": image_dino, "path": str(img_info["path"]), "2d_bbx": bbx2d,
            # GT infos
            "gt_corners": gt_corners,  # Normalized [0,1]
            "gt_depths": gt_depth_corners,  # Log space depths
            # padding mask
            "padding_mask": padding_mask,
            }
    


def build_image_dataloader(
    root_dir: Path,
    data_dir: Path,
    seed: int,
    split: str = "test",
    filter: bool = True, 
    batch_size: int = 8,
    da3_image_size: int = 448,
    dino_image_size: int = 512,
    target_quality: str = "Good",
    min_area: int = 32*32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader from a directory that only contains images.
    """
    json_paths = sorted(root_dir.glob(f"*{split}.json"))
    sampler = None
    if filter == True:
        json_list = [filtered_annotations(p, target_quality=target_quality, min_area=min_area, dino_size=dino_image_size) for p in json_paths]
    else:
        json_list = [json.loads(p.read_text(encoding="utf-8")) for p in json_paths]
    if split == "train":
        sampler = balanced_sampler(json_paths, json_list)
        shuffle = False
    dataset = AnnotationDataset(
        json_data=json_list,
        root_dir=data_dir,
        transform_da3=_default_transform(da3_image_size),
        transform_dino=_default_transform(dino_image_size),
        dino_image_size=dino_image_size,
    )
    
    generator = torch.Generator().manual_seed(seed)
    worker_init = functools.partial(_seed_worker, base_seed=seed)
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=worker_init,
    )


if __name__ == "__main__":
    # Example: iterate over a folder full of images
    sample_root = Path("/home/vmg/Desktop/layout2video/datasets/L2V_new")
    data_root = Path("/home/vmg/Desktop/layout2video/datasets")
    loader = build_image_dataloader(sample_root, data_root, split="val", filter=True, batch_size=2, da3_image_size=448, dino_image_size=512, num_workers=0)
    batch = next(iter(loader))
    print("Batch DA3 image tensor shape:", batch["image_da3"].shape)
    print("Batch DINO image tensor shape:", batch["image_dino"].shape)
    print("Batch file paths:", batch["path"])
    print("min/max DA3 image pixel values:", batch["image_da3"].min().item(), batch["image_da3"].max().item())
    print("min/max DINO image pixel values:", batch["image_dino"].min().item(), batch["image_dino"].max().item())
    print("length of dataset:",len(loader.dataset))
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import cv2

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


class ImageFolderDataset(Dataset):
    """
    Minimal dataset that collects every image file under a root directory.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
        transform_da3: Callable[[Image.Image], torch.Tensor],
        transform_dino: Callable[[Image.Image], torch.Tensor],
        recursive: bool = True,
    ) -> None:
        self.root = Path(root_dir).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not extensions:
            raise ValueError("extensions must contain at least one suffix.")
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.transform_da3 = transform_da3
        self.transform_dino = transform_dino
        self.paths = self._gather_files(recursive)
        if not self.paths:
            raise ValueError(f"No images with extensions {self.extensions} found in {self.root}")

    def _gather_files(self, recursive: bool) -> list[Path]:
        pattern = "**/*" if recursive else "*"
        files = [
            path
            for path in self.root.glob(pattern)
            if path.is_file() and path.suffix.lower() in self.extensions
        ]
        files.sort()
        return files

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        img_path = self.paths[index]
        image = Image.open(img_path).convert("RGB")
        image_da3 = self.transform_da3(image)
        image_dino = self.transform_dino(image)
        return {"image_da3": image_da3, "image_dino": image_dino, "path": str(img_path)}


def build_image_dataloader(
    data_dir: str | Path,
    *,
    batch_size: int = 8,
    da3_image_size: int = 448,
    dino_image_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    recursive: bool = True,
) -> DataLoader:
    """
    Create a DataLoader from a directory that only contains images.
    """
    dataset = ImageFolderDataset(
        data_dir,
        extensions=DEFAULT_EXTENSIONS,
        transform_da3=_default_transform(da3_image_size),
        transform_dino=_default_transform(dino_image_size),
        recursive=recursive,
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
    sample_root = Path("datasets/KITTI_object/testing/image_2")
    loader = build_image_dataloader(sample_root, batch_size=2, da3_image_size=448, dino_image_size=512, num_workers=0)
    batch = next(iter(loader))
    print("Batch DA3 image tensor shape:", batch["image_da3"].shape)
    print("Batch DINO image tensor shape:", batch["image_dino"].shape)
    print("Batch file paths:", batch["path"])
    print("min/max DA3 image pixel values:", batch["image_da3"].min().item(), batch["image_da3"].max().item())
    print("min/max DINO image pixel values:", batch["image_dino"].min().item(), batch["image_dino"].max().item())
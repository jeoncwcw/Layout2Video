# ...existing code...
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np



def extract_box_corners(sample: Dict[str, Any]) -> List[Tuple[float, float]]:
    # KEYS for labels fallback
    proj_corners = sample.get("projected_corners", [])
    uv_list: List[Tuple[float, float]] = list()
    
    for uv_dict in proj_corners:
        uv_list.append((uv_dict["u"], uv_dict["v"]))

    return uv_list

def extract_2d_bbox(sample: Dict[str, Any]) -> List[Tuple[float, float]]:
    bbox = sample.get("bbox2D_tight", [])
    if len(bbox) != 4:
        return []
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        


def draw_corners(image: np.ndarray, box_corners: List[Tuple[float, float]],
                color: Tuple[int, int, int] = (0, 255, 0),
                radius: int = 7, thickness: int = -1) -> np.ndarray:
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]

    for (u, v) in box_corners:
        x = int(round(u))
        y = int(round(v))

        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        cv2.circle(out, (x, y), radius, color, thickness)
    return out


def main() -> None:
    JSON_ROOT = Path("./datasets/L2V_new")
    datasets = [
        "nuScenes", "Objectron"
    ]
    splits = ["train", "val", "test"]
    DATASET_ROOT = Path("./datasets/")
    OUTPUT_DIR = Path("./test/sam2")
    
    for dataset in datasets:
        for split in splits:
            json_path = JSON_ROOT / f"{dataset}_{split}.json"
            output_dir = OUTPUT_DIR / f"{dataset}_{split}"
            # Load annotations
            with open(json_path, 'r') as f:
                data = json.load(f)
                image_info_map = {img["id"]: img for img in data["images"]}
            objects = data.get("annotations", [])

            random.shuffle(objects)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Visualize and save
            saved = 0
            for sample in objects:
                # quality = sample.get("quality")
                # if quality_count[quality] <= 0:
                #     continue
                # quality_count[quality] -= 1
                img = image_info_map.get(sample["image_id"])
                rel_path = img.get("file_path")
                img_path = DATASET_ROOT / rel_path
                if not img_path or not img_path.is_file():
                    raise FileNotFoundError(f"Image file not found: {img_path}")

                image = cv2.imread(str(img_path))
                if image is None:
                    raise RuntimeError(f"Failed to load image: {img_path}")

                # box_corners = extract_box_corners(sample)
                box_corners = extract_2d_bbox(sample)
                if not box_corners:
                    continue

                annotated = draw_corners(image, box_corners)
                # out_path = output_dir / f"{quality}{quality_count[quality]}_{img_path.stem}.jpg"
                out_path = output_dir / f"{img_path.stem}_vis.jpg"
                cv2.imwrite(str(out_path), annotated)
                saved += 1
                print(f"[INFO] saved {out_path}")

                # if saved >= count:
                #     break

            if saved == 0:
                print("No visualizations saved.")


if __name__ == "__main__":
    main()
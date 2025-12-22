from pathlib import Path
import numpy as np
from typing import List
import math
import torch
from torch.utils.data import WeightedRandomSampler

def balanced_sampler(json_paths: List[Path], target_quality: str, min_area: int, dino_size: int) -> WeightedRandomSampler:
    """
    Square Root Sampling
    Weight = 1 / sqrt(Count)
    """
    sample_dataset_indices = []
    dataset_counts = {}
    for json_file in json_paths:
        dataset_name = json_file.stem.split('_')[0]
        filtered_data = filtered_annotations(json_file, target_quality, min_area, dino_size)
        num_samples = len(filtered_data["annotations"])
        sample_dataset_indices.extend([dataset_name] * num_samples)
        dataset_counts[dataset_name] = num_samples
    
    print(f"[Sampler] Dataset Counts: {dataset_counts}")

    weights = []
    dataset_weights = {
        name: 1.0 / math.sqrt(count) for name, count in dataset_counts.items()
    }

    min_weight = min(dataset_weights.values())
    dataset_weights = {k: v / min_weight for k, v in dataset_weights.items()}

    for dataset_name in sample_dataset_indices:
        weights.append(dataset_weights[dataset_name])
    
    print(f"[Sampler] Dataset Weights: {dataset_weights}")

    for dataset_name in sample_dataset_indices:
        weights.append(dataset_weights[dataset_name])

    weights = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def filtered_annotations(json: Path, target_quality: str, min_area: int, dino_size: int) -> dict:
    with open(json, 'r') as f:
        data = json.load(f)
    
    annotations = data.get("annotations", [])
    image_map = {img["id"]: img for img in data.get("images", [])}
    
    filtered_annotations = []
    for obj in annotations:
        if obj.get("quality") != target_quality:
            continue
            
        bbox = obj.get("bbox2D_tight")
        if not bbox:
            continue
        if bbox[0] == -1 or bbox[1] == -1 or bbox[2] == -1 or bbox[3] == -1:
            continue
        
        img = image_map[obj["image_id"]]
        img_width, img_height = img["width"], img["height"]
        longest = max(img_width, img_height)
        scale = dino_size / float(longest)
        new_w, new_h = int(round(img_width * scale)), int(round(img_height * scale))
        
        pad_left, pad_top = (dino_size - new_w) // 2, (dino_size - new_h) // 2
        
        bbox = np.array(bbox, dtype=np.float32)
        bbox[0] = bbox[0] * scale + pad_left
        bbox[1] = bbox[1] * scale + pad_top
        bbox[2] = bbox[2] * scale + pad_left
        bbox[3] = bbox[3] * scale + pad_top
        
        box_width = bbox[2] - bbox[0]
        box_height = bbox[3] - bbox[1]
        box_area = box_width * box_height
        
        if box_area >= min_area:
            filtered_annotations.append(obj)
    
    data["annotations"] = filtered_annotations
    return data
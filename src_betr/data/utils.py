from pathlib import Path
import numpy as np
from typing import List
import math
import torch
from torch.utils.data import WeightedRandomSampler, Sampler
import torch.distributed as dist
import json

def balanced_sampler(json_paths: List[Path], json_data_list: List[dict],
                     is_ddp: bool=False, rank: int=0, world_size: int=1,
                     ) -> WeightedRandomSampler | Sampler:
    """
    Square Root Sampling
    Weight = 1 / sqrt(Count)
    """
    sample_dataset_indicies = []
    dataset_counts = {}
    for path, data in zip(json_paths, json_data_list):
        dataset_name = path.stem.split('_')[0]
        num_samples = len(data["annotations"])
        sample_dataset_indicies.extend([dataset_name] * num_samples)
        if num_samples > 0:
            dataset_counts[dataset_name] = num_samples
    
    print(f"[Sampler] Dataset Counts: {dataset_counts}")

    weights = []
    dataset_weights = {
        name: 1.0 / math.sqrt(count) for name, count in dataset_counts.items()
    }

    min_weight = min(dataset_weights.values())
    dataset_weights = {k: v / min_weight for k, v in dataset_weights.items()}
    
    print(f"[Sampler] Dataset Weights: {dataset_weights}")
    
    for dataset_name in sample_dataset_indicies:
        weights.append(dataset_weights[dataset_name])

    weights = torch.tensor(weights, dtype=torch.double)
    if is_ddp:
        return DistributedWeightedSampler(
            weights=weights,
            num_replicas=world_size,
            rank=rank,
            replacement=True
        )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def filtered_annotations(json_path: Path, target_quality: str, min_area: int, dino_size: int) -> dict:
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = data.get("annotations", [])
    image_map = {img["id"]: img for img in data.get("images", [])}
    
    filtered_annotations = []
    for obj in annotations:
        if obj.get("quality") != target_quality:
            continue
            
        bbox = obj.get("bbox2D_tight")
        if not bbox or -1 in bbox:
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

class DistributedWeightedSampler(Sampler):
    def __init__(self, weights, num_replicas=None, rank=None, replacement=True, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0
        
        total_len = len(self.weights)
        self.num_samples = math.ceil(total_len / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indicies = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        indicies = indicies[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        if not self.replacement and len(indicies) < self.num_samples:
             indicies += indicies[:(self.num_samples - len(indicies))]
        return iter(indicies)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch
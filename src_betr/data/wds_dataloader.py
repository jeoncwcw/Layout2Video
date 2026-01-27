import torch
import webdataset as wds
import math
from pathlib import Path
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
import random

DATASET_STATS_ANN_TRAIN = {
    "ARKitScenes": {"count": 0,      "type": "indoor"},
    "Hypersim":    {"count": 264154, "type": "indoor"},
    "KITTI":       {"count": 4435,   "type": "urban"},
    "nuScenes":    {"count": 47893,  "type": "urban"},
    "Objectron":   {"count": 28890,  "type": "general"},
    "SUNRGBD":     {"count": 13006,  "type": "indoor"},
}
DATASET_STATS_IMAGE_TRAIN = {
    "ARKitScenes": {"count": 0,      "type": "indoor"},
    "Hypersim":    {"count": 43852, "type": "indoor"},
    "KITTI":       {"count": 2045,   "type": "urban"},
    "nuScenes":    {"count": 18000,  "type": "urban"},
    "Objectron":   {"count": 25703,  "type": "general"},
    "SUNRGBD":     {"count": 3896,  "type": "indoor"},
}

class FeatureGeometryAug:
    def __init__(self, cfg):
        self.flip_prob = cfg.aug.flip_prob
        self.rot_range = cfg.aug.rot_range
        self.swap_map = [3, 2, 1, 0, 7, 6, 5, 4]
        self.cfg_noise = cfg.aug.feature_noise_sigma
        self.cfg_jitter = cfg.aug.box_jitter_sigma
        
        self.current_noise = 0.0
        self.current_jitter = 0.0
    
    def set_epoch(self, epoch):
        if epoch < 5:
            self.current_noise = 0.0
            self.current_jitter = 0.0
        else:
            self.current_noise = self.cfg_noise
            self.current_jitter = self.cfg_jitter
        
    def __call__(self, samples):
        for sample in samples:
            if self.rot_range > 0:
                angle = (torch.rand(1).item() * 2 - 1) * self.rot_range
                rad = math.radians(angle)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                
                # Feature rotation
                for key in ["feat_metric", "feat_mono", "feat_dino"]:
                    sample[key] = TF.rotate(sample[key], angle, interpolation=TF.InterpolationMode.BILINEAR, expand=False)
                # Corner rotation
                curr_corners = sample["gt_corners"] - 0.5
                new_corners = torch.empty_like(curr_corners)
                new_corners[:, 0] = curr_corners[:, 0] * cos_a - curr_corners[:, 1] * sin_a
                new_corners[:, 1] = curr_corners[:, 0] * sin_a + curr_corners[:, 1] * cos_a
                sample["gt_corners"] = (new_corners + 0.5).clamp(0, 1)
                
                # Box rotation
                x1, y1, x2, y2 = sample["2d_bbx"]
                bx = torch.tensor([x1, x2, x2, x1]) - 0.5
                by = torch.tensor([y1, y1, y2, y2]) - 0.5
                new_bx = bx * cos_a - by * sin_a + 0.5
                new_by = bx * sin_a + by * cos_a + 0.5
                sample["2d_bbx"] = torch.tensor([new_bx.min(), new_by.min(), new_bx.max(), new_by.max()]).clamp(0, 1)
                # padding rotation
                mask_float = sample["padding_mask"].float().unsqueeze(0) # [1, H, W]
                rotated_mask = TF.rotate(mask_float, angle, interpolation=TF.InterpolationMode.NEAREST, expand=False, fill=1.0)
                sample["padding_mask"] = rotated_mask.squeeze(0) > 0.5
                
            if torch.rand(1) < self.flip_prob:
                for key in ["feat_metric", "feat_mono", "feat_dino", "padding_mask"]:
                    sample[key] = TF.hflip(sample[key])
                
                sample["gt_corners"][:, 0] = 1.0 - sample["gt_corners"][:, 0].clamp(0, 1)
                sample["gt_corners"] = sample["gt_corners"][self.swap_map]
                x1, y1, x2, y2 = sample["2d_bbx"]
                sample["2d_bbx"] = torch.tensor([1.0 - x2, y1, 1.0 - x1, y2])
            if self.current_noise > 0:
                for key in ["feat_metric", "feat_mono", "feat_dino"]:
                    sample[key] += torch.randn_like(sample[key]) * self.current_noise
            if self.current_jitter > 0:
                noise = torch.randn_like(sample["2d_bbx"]) * self.current_jitter
                sample["2d_bbx"] = torch.clamp(sample["2d_bbx"] + noise, 0, 1)
            yield sample
        
class FlattenSamples:
    def __init__(self, dino_img_size: int = 512, max_cap: int = 4):
        self.dino_img_size = dino_img_size
        self.max_cap = max_cap
    def __call__(self, samples):
        for sample in samples:
            if not sample["targets.pth"]:
                continue
            features = sample["feat.pth"]
            pad_info = sample["pad_info.pth"]
            
            pad_top, pad_left, new_h, new_w = pad_info
            padding_mask = torch.ones((self.dino_img_size, self.dino_img_size), dtype=torch.bool)
            padding_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = False
        
            f_metric = features["metric"].float().squeeze(0)
            f_mono = features["mono"].float().squeeze(0)
            f_dino = features["dino"].float().squeeze(0)
            if len(sample["targets.pth"]) > self.max_cap:
                selected_targets = random.sample(sample["targets.pth"], self.max_cap)
            else:
                selected_targets = sample["targets.pth"]
            for target in selected_targets:
                yield {
                    "feat_metric": f_metric,
                    "feat_mono": f_mono,
                    "feat_dino": f_dino,
                    "2d_bbx": target["2d_bbx"],
                    "gt_corners": target["gt_corners"],
                    "gt_depths": target["gt_depths"],
                    "padding_mask": padding_mask,
                }     
            
def get_balanced_weights(found_datasets, max_cap=4):
    image_stats = DATASET_STATS_IMAGE_TRAIN
    ann_stats = DATASET_STATS_ANN_TRAIN
    raw_scores = {}
    for name in found_datasets:
        key = next((k for k in image_stats if k in name), None)
        if key:
            img_count = image_stats[key]["count"]
            ann_count = ann_stats[key]["count"]
            
            avg_ann = ann_count / max(img_count, 1)
            expected_yield = min(avg_ann, max_cap)
            score = math.sqrt(img_count) / max(expected_yield, 0.5)
            raw_scores[name] = score
    total_score = sum(raw_scores.values())
    final_weights = {k: v / total_score for k, v in raw_scores.items()}
    
    return final_weights

    
def build_wds_feature_dataloader(
    cfg: any,
    wds_root: Path,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    dino_img_size: int = 512,
    epoch_length: int = 50000,
    world_size: int = 1
):
    wds_root = Path(wds_root)
    dataset_dirs = sorted(list(wds_root.glob(f"*_{split}")))
    
    if not dataset_dirs:
        raise ValueError(f"No WDS dataset found in {wds_root} for split {split}")
    found_dataset_names = [d.name for d in dataset_dirs]
    
    weight_map = get_balanced_weights(found_dataset_names)
    print(f"WDS Dataloader - Using datasets and weights: {weight_map}")
    urls = []
    weights = []    
    for d_dir in dataset_dirs:
        d_name = d_dir.name
        if d_name not in weight_map:
            continue
        shards = sorted(list(d_dir.glob("shard-*.tar")))
        if not shards: raise ValueError(f"No shards found in {d_dir}")
        if len(shards) == 1: url = str(shards[0])
        else:
            last_shard_idx = int(shards[-1].stem.split('-')[1])
            url = str(d_dir / f"shard-{{000000..{last_shard_idx:06d}}}.tar")
        urls.append(url)
        w = weight_map[d_name]
        weights.append(w)
        
    sum_w = sum(weights)
    weights = [w / sum_w for w in weights]
    
    transform = FlattenSamples(dino_img_size=dino_img_size, max_cap=4)
    aug_transform = None
    if split == "train":
        aug_transform = FeatureGeometryAug(cfg)
        
    datasets = []
    for url in urls:
        ds = (wds.WebDataset(url, nodesplitter=wds.split_by_node, shardshuffle=1000, empty_check=False)
        .repeat()
        .shuffle(500)
        .decode("torch")
        .compose(transform)
        .shuffle(200)
        )
        if aug_transform is not None:
            ds = ds.compose(aug_transform)
        datasets.append(ds)
    
    if len(datasets) > 1:
        dataset = wds.RandomMix(datasets, weights)
    else:
        dataset = datasets[0]
    batches_per_rank = epoch_length // (batch_size * world_size)
        
    loader = (
        wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers,
                      persistent_workers=True, pin_memory=True)
        .batched(batch_size, partial=False)
        .with_epoch(batches_per_rank)
    )
    return loader, aug_transform

def count_wds_samples(wds_root: Path, split: str = "train") -> dict:
    # Calculate number of samples and annotations in WDS dataset
    import tarfile
    
    wds_root = Path(wds_root)
    dataset_dirs = sorted(list(wds_root.glob(f"*_{split}")))
    
    counts = {}
    total_annotations = 0
    
    for d_dir in dataset_dirs:
        d_name = d_dir.name
        shards = sorted(list(d_dir.glob("shard-*.tar")))
        
        sample_count = 0
        annotation_count = 0
        
        for shard_path in shards:
            with tarfile.open(shard_path, 'r') as tar:
                # Each sample is distinguished by __key__
                keys = set()
                for member in tar.getmembers():
                    key = member.name.rsplit('.', 1)[0]
                    keys.add(key)
                sample_count += len(keys)
                
                # Calculate number of annotations from targets.pth files
                for member in tar.getmembers():
                    if member.name.endswith('targets.pth'):
                        f = tar.extractfile(member)
                        targets = torch.load(f, map_location='cpu')
                        annotation_count += len(targets)
        
        counts[d_name] = {
            "samples": sample_count,
            "annotations": annotation_count
        }
        total_annotations += annotation_count
    
    counts["_total"] = {"annotations": total_annotations}
    return counts

if __name__ == "__main__":
    
    print("\n" + "=" * 50)
    print("📊 Testing build_wds_feature_dataloader()")
    print("=" * 50)
    
    wds_root = Path("/home/vmg/Desktop/layout2video/datasets/betr_wds")
    cfg = OmegaConf.load("/home/vmg/Desktop/layout2video/src_betr/configs/betr_config.yaml")
    if not wds_root.exists():
        print(f"⚠️  WDS root not found: {wds_root}")
    else:
        print("\n📊 Counting actual samples...")
        counts = count_wds_samples(wds_root, split="train")
        for name, info in counts.items():
            print(f"  {name}: {info}")
        try:
            loader = build_wds_feature_dataloader(
                cfg = cfg,
                wds_root=wds_root,
                split="train",
                batch_size=4,
                num_workers=2,
                epoch_length=10,
            )
            
            print("\n🔄 Checking batches...")
            for i, batch in enumerate(loader):
                print(f"\nBatch {i}:")
                for k, v in batch.items():
                    shape = v.shape if hasattr(v, 'shape') else len(v)
                    print(f"  {k}: {shape}")
                if i >= 1:
                    break
            print("\n✅ Done!")
        except Exception as e:
            print(f"❌ Error: {e}")
        
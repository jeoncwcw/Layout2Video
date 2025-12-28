import torch
import webdataset as wds
import math
from pathlib import Path
from torch.utils.data import DataLoader

DATASET_STATS = {
    "ARKitScenes": {"count": 0,      "type": "indoor"},
    "Hypersim":    {"count": 264154, "type": "indoor"},
    "KITTI":       {"count": 4435,   "type": "outdoor"},
    "nuScenes":    {"count": 47893,  "type": "outdoor"},
    "Objectron":   {"count": 28890,  "type": "indoor"},
    "SUNRGBD":     {"count": 13006,  "type": "indoor"},
}

def flatten_samples(samples, dino_img_size=512):
    for sample in samples:
        features = sample["feat.pth"]
        targets_list = sample["targets.pth"]
        pad_info = sample["pad_info.pth"]
        
        pad_top, pad_left, new_h, new_w = pad_info
        padding_mask = torch.ones((dino_img_size, dino_img_size), dtype=torch.bool)
        padding_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = False
        
        f_metric = features["metric"].float().squeeze(0)
        f_mono = features["mono"].float().squeeze(0)
        f_dino = features["dino"].float().squeeze(0)
        
        for target in targets_list:
            yield {
                "feat_metric": f_metric,
                "feat_mono": f_mono,
                "feat_dino": f_dino,
                
                "2d_bbx": target["2d_bbx"],
                "gt_center": target["gt_center"],
                "gt_offsets_3d": target["gt_offsets_3d"],
                "gt_corners_3d": target["gt_corners_3d"],
                "gt_center_depth": target["gt_center_depth"],
                "gt_depth_offsets": target["gt_depth_offsets"],
                "gt_metric_depth": target["gt_metric_depth"],
                "padding_mask": padding_mask,
            }
            
def get_hierarchical_weights(found_datasets):
    groups = {"indoor": [], "outdoor": []}
    
    for name in found_datasets:
        key = next((k for k in DATASET_STATS if k in name), None)
        if key:
            groups[DATASET_STATS[key]["type"]].append((name, DATASET_STATS[key]["count"]))
    
    final_weights = {}
    group_probs = {"indoor": 0.6, "outdoor": 0.4}
    
    for g_name, datasets in groups.items():
        datasets = [(name, count) for name, count in datasets if count > 0]
        if not datasets: continue
        
        raw_weights = [1.0 / math.sqrt(count) for _, count in datasets]
        total_score = sum(raw_weights)
        
        target_prob = group_probs[g_name]
        for (d_name, _), w in zip(datasets, raw_weights):
            final_weights[d_name] = (w / total_score) * target_prob
    
    return final_weights
    
def build_wds_feature_dataloader(
    wds_root: Path,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    dino_img_size: int = 512,
    epoch_length: int = 50000,
):
    wds_root = Path(wds_root)
    dataset_dirs = sorted(list(wds_root.glob(f"*_{split}")))
    
    if not dataset_dirs:
        raise ValueError(f"No WDS dataset found in {wds_root} for split {split}")
    
    found_dataset_names = [d.name for d in dataset_dirs]
    
    if split == "train":
        weight_map = get_hierarchical_weights(found_dataset_names)
    else:
        weight_map = {name: 1.0 for name in found_dataset_names}
        
    urls = []
    weights = []
    
    print(f"\n📊 [{split}] Sampling Strategy:")
    print(f"{'Dataset':<20} | {'Type':<8} | {'Count':<8} | {'Mix Weight':<10}")
    print("-" * 55)
        
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
        base_name = next((k for k in DATASET_STATS if k in d_name), "Unknown")
        d_type = DATASET_STATS[base_name]["type"]
        d_count = DATASET_STATS[base_name]["count"]
        print(f"{d_name:<20} | {d_type:<8} | {d_count:<8} | {w:.4f}")
        
    sum_w = sum(weights)
    weights = [w / sum_w for w in weights]
    
    datasets = []
    for url in urls:
        ds = (wds.WebDataset(url, nodesplitter=wds.split_by_node, shardshuffle=True)
        .shuffle(1000)
        .decode("torch")
        .compose(lambda src: flatten_samples(src, dino_img_size)))
        datasets.append(ds)
    
    if len(datasets) > 1:
        dataset = wds.RandomMix(datasets, weights)
    else:
        dataset = datasets[0]
        
    loader = (
        wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers)
        .batched(batch_size, partial=False)
        .with_epoch(epoch_length // batch_size)
    )
    
    return loader

if __name__ == "__main__":
    # 1) 가중치 계산 테스트
    print("=" * 50)
    print("📊 Testing get_hierarchical_weights()")
    print("=" * 50)
    
    fake_datasets = ["Hypersim_train", "KITTI_train", "nuScenes_train", "SUNRGBD_train", "Objectron_train"]
    weights = get_hierarchical_weights(fake_datasets)
    
    for name, w in weights.items():
        print(f"  {name}: {w:.4f}")
    print(f"  Total: {sum(weights.values()):.4f}")
    
    # 2) 실제 데이터로더 테스트 (경로 수정 필요)
    print("\n" + "=" * 50)
    print("📊 Testing build_wds_feature_dataloader()")
    print("=" * 50)
    
    wds_root = Path("/home/vmg/Desktop/layout2video/datasets/betr_wds")
    
    if not wds_root.exists():
        print(f"⚠️  WDS root not found: {wds_root}")
    else:
        try:
            loader = build_wds_feature_dataloader(
                wds_root=wds_root,
                split="train",
                batch_size=4,
                num_workers=0,
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
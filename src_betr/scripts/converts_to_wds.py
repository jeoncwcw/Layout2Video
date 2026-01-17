from collections import defaultdict
import json
import torch
import numpy as np
import webdataset as wds
from pathlib import Path
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src_betr.data.utils import filtered_annotations

def process_sample(ann_data, pad_info, dino_img_size=512):
    pad_top, pad_left, new_h, new_w, scale = pad_info
    
    bbx2d_orig = np.array(ann_data["bbox2D_tight"], dtype=np.float32)
    bbx2d_processed = bbx2d_orig * scale
    bbx2d_processed[[0, 2]] += pad_left
    bbx2d_processed[[1, 3]] += pad_top
    bbx2d_processed = torch.tensor(bbx2d_processed, dtype=torch.float32) / dino_img_size

    # 3D bb8 processing
    corners_list = ann_data["projected_corners"]
    coords = [(float(c["u"]), float(c["v"])) for c in corners_list]
    bbx3d_bb8 = torch.tensor(coords, dtype=torch.float32)
    
    bbx3d_bb8 = bbx3d_bb8 * scale
    bbx3d_bb8[:, 0] += pad_left
    bbx3d_bb8[:, 1] += pad_top
    bbx3d_bb8 = bbx3d_bb8 / dino_img_size # Normalize to [0,1] range
    bbx3d_center = bbx3d_bb8.mean(dim=0) # [2]
    offsets_3d = bbx3d_bb8 - bbx3d_center.unsqueeze(0) # Normalized offsets
    offsets_3d = offsets_3d.flatten()  # [16]

    # depth (Metric to Stable Monodepth Style)
    raw_depths = torch.clamp(torch.tensor(ann_data["depth"], dtype=torch.float32), min=1e-3) # [8]
    center_depth = raw_depths.mean()
    box_w, box_h = (bbx2d_processed[2] - bbx2d_processed[0]) * dino_img_size, (bbx2d_processed[3] - bbx2d_processed[1]) * dino_img_size
    box_scale = torch.sqrt(box_w**2 + box_h**2)
    gt_canonical_depth = torch.log(center_depth * box_scale + 1e-8) # log space canonical depth
    depth_offsets = torch.log(raw_depths) - torch.log(center_depth) # Log space offsets (not need to normalize)
    
        

    return {
        "2d_bbx": bbx2d_processed,
        "gt_center": bbx3d_center,
        "gt_offsets_3d": offsets_3d,
        "gt_corners_3d": bbx3d_bb8,
        "gt_center_depth": gt_canonical_depth,
        "gt_depth_offsets": depth_offsets,
        "gt_metric_depth": center_depth,
    }
    
def main():
    json_root = Path(PROJECT_ROOT / "datasets" / "L2V_ordered")
    feature_root = Path("./datasets/betr_features")
    output_root = Path("./datasets/betr_wds")
    output_root.mkdir(parents=True, exist_ok=True)
    image_map_path = Path("./datasets/image_map.json")
    dino_img_size = 512
    
    with open(image_map_path, 'r') as f:
        image_map = json.load(f)
    split = "train"
    json_paths = sorted(json_root.glob(f"*{split}.json"))
    
    for json_path in json_paths:
        dataset_name = json_path.stem.replace(f"_{split}", "")
        print(f"Processing dataset: {dataset_name} ({split})")
        data = filtered_annotations(json_path)
        img2anns = defaultdict(list)
        for ann in data["annotations"]:
            img2anns[ann["image_id"]].append(ann)
        img_info_map = {img["id"]: img for img in data["images"]}
        # Shard setting
        save_subdir = output_root / f"{dataset_name}_{split}"
        save_subdir.mkdir(parents=True, exist_ok=True)
        pattern = str(save_subdir / "shard-%06d.tar")
        
        with wds.ShardWriter(pattern, maxcount=1000, maxsize=3e9) as sink:
            for img_id, anns in tqdm(img2anns.items(), desc=f"Processing {dataset_name}"):
                img_info = img_info_map[img_id]
                rel_path = img_info["file_path"]
                if rel_path not in image_map:
                    raise ValueError(f"Image path {rel_path} not found in image map.")
                feat_path = feature_root / image_map[rel_path]
                if not feat_path.exists():
                    raise ValueError(f"Feature path {feat_path} does not exist.")
                features = torch.load(feat_path, map_location='cpu')
                # padding infos
                orig_w, orig_h = img_info["width"], img_info["height"]
                longest = max(orig_w, orig_h)
                scale = dino_img_size / float(longest)
                new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
                pad_left, pad_top = (dino_img_size - new_w) // 2, (dino_img_size - new_h) // 2
                pad_info = (pad_top, pad_left, new_h, new_w, scale)
                pad_info_save = torch.tensor([pad_top, pad_left, new_h, new_w], dtype=torch.int16)
                # Ann process and save
                targets_list = [process_sample(ann, pad_info, dino_img_size) for ann in anns]
                key = rel_path.replace("/", "_").replace(".", "_")
                sample = {
                    "__key__": key,
                    "feat.pth": features,
                    "targets.pth": targets_list,
                    "pad_info.pth": pad_info_save
                }
                try:
                    sink.write(sample)
                except Exception as e:
                    print(f"Error writing sample {key}: {e}")
                else:
                    feat_path.unlink()  # Delete feature file after successful write
    print("All datasets processed and saved to WDS format.")
    
if __name__ == "__main__":
    main()
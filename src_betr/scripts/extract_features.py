import os
import json
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
import hashlib
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.feature_modules import DA3FeatureExtractor, DINOv3FeatureExtractor
from data.image_dataloader import _default_transform
from data.utils import filtered_annotations

def get_filtered_unique_images(root_json_dir, target_quality="Good", min_area=32*32, dino_size=512):
    unique_paths = set()
    
    json_paths = list(Path(root_json_dir).glob("*train.json"))
    for path in json_paths:
        filtered_data = filtered_annotations(path, target_quality, min_area, dino_size)
        valid_image_ids = [ann["image_id"] for ann in filtered_data["annotations"]]
        image_map = {img["id"]: img["file_path"] for img in filtered_data["images"]}
        for img_id in valid_image_ids:
            if img_id in image_map:
                img_path = image_map[img_id]
                unique_paths.add(img_path)
    
    print(f"Total unique filtered images: {len(unique_paths)}")
    return sorted(list(unique_paths))

def worker(rank, world_size, all_image_rel_paths, cfg, save_root):
    my_chunk = all_image_rel_paths[rank::world_size]
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    feat_dir = save_root / "betr_features"
    
    metric_ext = DA3FeatureExtractor(cfg_path=cfg.metricdepth_cfg_path, checkpoint_path=cfg.metricdepth_checkpoint_path, device=device)
    mono_ext = DA3FeatureExtractor(cfg_path=cfg.monodepth_cfg_path, checkpoint_path=cfg.monodepth_checkpoint_path, device=device)
    dino_ext = DINOv3FeatureExtractor(checkpoint_path=cfg.dinov3_checkpoint_path, device=device)
    
    transform_da3 = _default_transform(cfg.data.da3_image_size)
    transform_dino = _default_transform(cfg.data.dino_image_size)
    
    local_mapping = {}
    pbar = tqdm(my_chunk, desc=f"[GPU {rank}]", position=rank, leave=True)
    
    with torch.inference_mode():    
        for rel_path in pbar:
            full_path = save_root / rel_path
            if not full_path.exists():
                pbar.write(f"Image not found: {full_path}")
                continue
            
            
            path_hash = hashlib.md5(str(rel_path).encode()).hexdigest()
            feat_filename = feat_dir / f"feat_{path_hash}.pth"
            if not feat_filename.exists():
                img = Image.open(full_path).convert("RGB")
                img_da3 = transform_da3(img).unsqueeze(0).to(device)
                img_dino = transform_dino(img).unsqueeze(0).to(device)
            
                f_metric = metric_ext(img_da3).to(torch.bfloat16).cpu()
                f_mono = mono_ext(img_da3).to(torch.bfloat16).cpu()
                f_dino = dino_ext(img_dino).to(torch.bfloat16).cpu()
                torch.save({
                    "metric": f_metric,
                    "mono": f_mono,
                    "dino": f_dino,
                }, feat_filename)
                local_mapping[str(rel_path)] = str(f"feat_{path_hash}.pth")
            
    with open(save_root / f"image_map_rank{rank}.json", "w") as f:
        json.dump(local_mapping, f, indent=4)
        
def extract_features_parallel():
    cfg = OmegaConf.load("./src_betr/configs/betr_config.yaml")
    save_root = Path("/home/vmg/Desktop/layout2video/datasets")
    (save_root / "betr_features").mkdir(parents=True, exist_ok=True)
    image_map_path = save_root / "image_map.json"
    
    image_rel_paths = get_filtered_unique_images(
        root_json_dir=str(save_root / "L2V_new"),
    )
    
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for feature extraction.")
    
    
    mp.spawn(worker, args=(world_size, image_rel_paths, cfg, save_root), nprocs=world_size, join=True)
    
    final_mapping = {}
    if image_map_path.exists():
        with open(image_map_path, "r") as f:
            final_mapping = json.load(f)
            
    for rank in range(world_size):
        map_path = save_root / f"image_map_rank{rank}.json"
        with open(map_path, "r") as f:
            final_mapping.update(json.load(f))
        os.remove(map_path)
    
    with open(image_map_path, "w") as f:
        json.dump(final_mapping, f, indent=4)
    print(f"Feature extraction completed and mapped to image_map.json")
    
if __name__ == "__main__":
    extract_features_parallel()
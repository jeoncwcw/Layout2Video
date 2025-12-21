import json
from pathlib import Path
from collections import defaultdict
import os

def main():
    datasets = [
        "ARKitScenes", "Hypersim", "KITTI",
        "nuScenes", "Objectron", "SUNRGBD"
    ]
    splits = ["train", "val", "test"]
    
    MIN_SIZE = 32
    MIN_AREA = MIN_SIZE * MIN_SIZE 
    TARGET_QUALITY = "Good"
    
    dataset_root = Path("./datasets/L2V")
    
    print(f"Filtering Criteria: Quality='{TARGET_QUALITY}', Min Area>={MIN_AREA}px ({MIN_SIZE}x{MIN_SIZE})")
    print("="*60)
    print(f"{'Dataset':<15} | {'Split':<6} | {'Total Objs':<10} | {'Filtered':<10} | {'Survival %':<10}")
    print("-" * 60)

    grand_total_before = 0
    grand_total_after = 0

    for dataset in datasets:
        for split in splits:
            json_file = dataset_root / f"{dataset}_{split}.json"
            
            if not json_file.exists():
                continue
                
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                annotations = data.get("annotations", [])
                total_count = len(annotations)
                filtered_count = 0
                
                for obj in annotations:
                    if obj.get("quality") != TARGET_QUALITY:
                        continue
                        
                    bbox = obj.get("bbox2D_tight")
                    if not bbox:
                        continue
                        
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    if area >= MIN_AREA:
                        filtered_count += 1
                
                ratio = (filtered_count / total_count * 100) if total_count > 0 else 0
                print(f"{dataset:<15} | {split:<6} | {total_count:<10} | {filtered_count:<10} | {ratio:>9.1f}%")
                
                if split == "train":
                    grand_total_before += total_count
                    grand_total_after += filtered_count

            except Exception as e:
                print(f"[ERROR] Processing {json_file}: {e}")
        
        print("-" * 60)

    print("="*60)
    print(f"🏆 Total Train Objects (Filtered): {grand_total_after} / {grand_total_before}")
    print("Done! 🥰")

if __name__ == "__main__":
    main()
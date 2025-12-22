import json
from pathlib import Path
import numpy as np

def main():
    datasets = [
        "ARKitScenes", "Hypersim", "KITTI",
        "nuScenes", "Objectron", "SUNRGBD"
    ]
    splits = ["train", "val", "test"]
    
    MIN_SIZE = 32
    MIN_AREA = MIN_SIZE * MIN_SIZE 
    TARGET_QUALITY = "Good"
    
    dataset_root = Path("./datasets/L2V_new")
    
    print(f"Filtering Criteria: Quality='{TARGET_QUALITY}', Min Area>={MIN_AREA}px ({MIN_SIZE}x{MIN_SIZE})")
    # print(f"Weird Boxes Criterion: Any bbox coordinate is -1")
    print("="*60)
    print(f"{'Dataset':<15} | {'Split':<6} | {'Total Objs':<10} | {'Filtered':<10} | {'Survival %':<10}")
    # print(f"{'Dataset':<15} | {'Split':<6} | {'Total Objs':<10} | {'Wierd Boxes':<10}")
    print("-" * 60)

    grand_total_before = 0
    grand_total_after = 0

    for dataset in datasets:
        for split in splits:
            json_file = dataset_root / f"{dataset}_{split}.json"
            
                
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                annotations = data.get("annotations", [])
                total_count = len(annotations)
                filtered_count = 0
                # count_weird_boxes = 0
                image_map = {img["id"]: img for img in data.get("images", [])}
                
                for obj in annotations:
                    if obj.get("quality") != TARGET_QUALITY:
                        continue
                        
                    bbox = obj.get("bbox2D_tight")
                    if not bbox:
                        continue
                    if bbox[0] == -1 or bbox[1] == -1 or bbox[2] == -1 or bbox[3] == -1:
                        continue
                    
                    img = image_map[obj["image_id"]]
                    img_width, img_height = img["width"], img["height"]
                    longest = max(img_width, img_height)
                    scale = 512 / float(longest)
                    new_w, new_h = int(round(img_width * scale)), int(round(img_height * scale))
                    pad_left, pad_top = (512 - new_w) // 2, (512 - new_h) // 2
                    
                    bbox = np.array(bbox, dtype=np.float32)
                    bbox_processed = bbox * scale
                    bbox_processed[[0, 2]] += pad_left
                    bbox_processed[[1, 3]] += pad_top
                    x1, y1, x2, y2 = bbox_processed
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width <=0 or height <=0:
                        continue
                    area = width * height
                    
                    if area >= MIN_AREA:
                        filtered_count += 1
                
                ratio = (filtered_count / total_count * 100) if total_count > 0 else 0
                print(f"{dataset:<15} | {split:<6} | {total_count:<10} | {filtered_count:<10} | {ratio:>9.1f}%")
                # print(f"{dataset:<15} | {split:<6} | {total_count:<10} | {count_weird_boxes:<10}")                
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
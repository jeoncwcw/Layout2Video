import argparse
import json
from collections import defaultdict
import os
from typing import List

def get_json_file_paths(json_dir: str, split: str) -> List[str]:
    json_files = []
    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith(f"{split}.json"):
                json_files.append(os.path.join(root, file))
    return json_files

def main():
    parser = argparse.ArgumentParser(description="Get number of objects per image from JSON annotations.")
    parser.add_argument("--json_dir", type=str, required=True, help="Path to the JSON annotation directory.")
    args = parser.parse_args()
    json_dir = args.json_dir
    split_list = ["train", "val", "test"]
    res_dict = {}
    for split in split_list:
        json_files = get_json_file_paths(json_dir, split)
        
        objs_num_list = defaultdict(int)
        for json_path in json_files:
            print("="*10+f"Processing..{json_path}"+"="*10)
            with open(json_path, 'r') as f:
                data = json.load(f)
                image_info_map = {img["id"]: img for img in data["images"]}
            objects = data.get("annotations", [])
            objs_per_image = {}
            for obj in objects:
                image_id = obj["image_id"]
                if image_id not in objs_per_image:
                    objs_per_image[image_id] = 0
                objs_per_image[image_id] += 1
            
            for img_id, num_objs in objs_per_image.items():
                objs_num_list[num_objs] += 1
                if num_objs == 2046:
                    img = image_info_map.get(img_id)
                    img_path = img.get("file_path")
                    print(f"Image with 2046 objects: {img_path}")
            
            print(f"Processed {json_path}")
        out = dict(sorted(objs_num_list.items()))
        result = {"1": 0, "2": 0, "3": 0, "4": 0, "5-10": 0, "11-20": 0, "21-50": 0, "51-100": 0, "101-200": 0, "201-500": 0, "501-1000": 0, ">1000": 0}
        for num in out.keys():
            if num == 1:
                result["1"] += out[num]
            elif num == 2:
                result["2"] += out[num]
            elif num == 3:
                result["3"] += out[num]
            elif num == 4:
                result["4"] += out[num]
            elif 5 <= num <= 10:
                result["5-10"] += out[num]
            elif 11 <= num <= 20:
                result["11-20"] += out[num]
            elif 21 <= num <= 50:
                result["21-50"] += out[num]
            elif 51 <= num <= 100:
                result["51-100"] += out[num]
            elif 101 <= num <= 200:
                result["101-200"] += out[num]
            elif 201 <= num <= 500:
                result["201-500"] += out[num]
            elif 501 <= num <= 1000:
                result["501-1000"] += out[num]
            else:
                result[">1000"] += out[num]
        res_dict[split] = result
        
    print(f"\t\t===Train===\t=== Val ===\t===Test===")
    for k in res_dict["train"].keys():
        if k == "1" or k == "2" or k =="3" or k == "4" or k == "5-10":
            print(f"[{k}]:\t\t  {res_dict['train'][k]}\t\t  {res_dict['val'][k]}\t\t  {res_dict['test'][k]}")
        else: 
            print(f"[{k}]:\t  {res_dict['train'][k]}\t\t  {res_dict['val'][k]}\t\t  {res_dict['test'][k]}")
if __name__ == "__main__":
    main()
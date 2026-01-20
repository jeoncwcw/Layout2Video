import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def sort_corners_2d_spatial(projected_corners, depths):
    combined = []
    for i in range(len(projected_corners)):
        combined.append({
            "u": float(projected_corners[i]["u"]),
            "v": float(projected_corners[i]["v"]),
            "depth": float(depths[i])
        })
    v_sorted = sorted(combined, key=lambda x: x["v"], reverse=True)
    
    bottom_4 = v_sorted[:4]
    top_4 = v_sorted[4:]
    
    bottom_4_final = sorted(bottom_4, key=lambda x: x["u"])
    top_4_final = sorted(top_4, key=lambda x: x["u"])
    final_list = bottom_4_final + top_4_final
    
    new_corners = [{"u": p["u"], "v": p["v"]} for p in final_list]
    new_depths = [p["depth"] for p in final_list]
    
    return new_corners, new_depths

def process_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = list(input_path.glob("*.json"))
    
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for obj in data.get("annotations", []):
            corners = obj.get("projected_corners", [])
            depths = obj.get("depth", [])
            
            if len(corners) == 8 and len(depths) == 8:
                new_corners, new_depths = sort_corners_2d_spatial(corners, depths)
                obj["projected_corners"] = new_corners
                obj["depth"] = new_depths
    
        save_path = output_path / json_file.name
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
if __name__ == "__main__":
    input_dir = "./datasets/L2V_ordered"
    output_dir = "./datasets/L2V_2d_ordered"
    process_dataset(input_dir, output_dir)
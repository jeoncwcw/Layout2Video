import argparse
import json
from typing import Dict, List
from pathlib import Path
import os


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--src_json", type=Path, required=False, default="./datasets/L2V_new",
    )
    argparser.add_argument(
        "--dst_json", type=Path, required=False, default="./datasets/L2V_ordered",
    )
    args = argparser.parse_args()
    src_json = args.src_json
    dst_json = args.dst_json
    
    with open(src_json) as f:
        src_data = json.load(f)
    with open(dst_json) as f:
        dst_data = json.load(f)
        
    dst_anns = dst_data["annotations"]
    src_ann_map = {ann["id"]: ann for ann in src_data["annotations"]}
    dst_ann_map = {ann["id"]: ann for ann in dst_data["annotations"]}
    
    for ann in dst_anns:
        ann_id = ann["id"]
        if ann_id in src_ann_map:
            src_ann = src_ann_map[ann_id]
            dst_ann = dst_ann_map[ann_id]
            dst_ann["bbox2D_tight"] = src_ann["bbox2D_tight"]
        else:
            print(f"Warning: Annotation ID {ann_id} not found in source JSON.")
    
    with open(dst_json, "w") as f:
        json.dump(dst_data, f, indent=4)
        
if __name__ == "__main__":
    main()
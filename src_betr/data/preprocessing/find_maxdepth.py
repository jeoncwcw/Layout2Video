import json
import math
from pathlib import Path


if __name__ == "__main__":
    ROOT_json = Path(__file__).resolve().parents[3] / "datasets" / "L2V"
    json_paths = list(ROOT_json.glob("**/*.json"))

    max_depth = float("-inf")
    total = 0.0
    total_sq = 0.0
    count = 0

    for json_path in json_paths:
        with open(json_path, "r") as f:
            data = json.load(f)

        for obj in data.get("annotations", []):
            depth_list = obj.get("depth")
            if not depth_list:
                continue
            for depth in depth_list:
                max_depth = max(max_depth, depth)
                total += depth
                total_sq += depth * depth
                count += 1

    if count > 0:
        mean_depth = total / count
        variance = total_sq / count - mean_depth ** 2
        std_depth = math.sqrt(max(variance, 0.0))
    else:
        mean_depth = float("nan")
        std_depth = float("nan")
        max_depth = "N/A"

    print(f"Files scanned: {len(json_paths)}")
    print(f"Depth count: {count}")
    print(f"Maximum depth: {max_depth}")
    print(f"Mean depth: {mean_depth}")
    print(f"Std depth: {std_depth}")
import numpy as np
import argparse
import json

def find_basis_points(corners_3d):
    origin_idx = 0
    origin = corners_3d[origin_idx]
    dists = np.linalg.norm(corners_3d - origin, axis=1)
    neighbor_indices = np.argsort(dists)[1:4]
    basis_indices = [origin_idx] + neighbor_indices.tolist()
    
    return basis_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find basis points from 3D corners.")
    parser.add_argument("--input_file", type=str, help="Path to the input .npy file containing 3D corners.")
    args = parser.parse_args()
    print("hello")
    
    with open(args.input_file, "r") as f:
        data = json.load(f)
        objects = data["annotations"]
        print(f"Total objects: {len(objects)}")
        
    for i, obj in enumerate(objects):
        corners_3d = np.array(obj["bbox3D_cam"])
        basis_indices = find_basis_points(corners_3d)
        print(f"Object {i} basis point indices: {basis_indices}")
        if i >= 10:
            break
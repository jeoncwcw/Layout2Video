import json

datasets = [
    "ARKitScenes", "Hypersim", "KITTI",
    "nuScenes", "Objectron", "SUNRGBD",
]

for dataset in datasets:
    train_path  = f"./datasets/L2V/{dataset}_train.json"
    test_path   = f"./datasets/L2V/{dataset}_test.json"
    val_path = f"./datasets/L2V/{dataset}_val.json"
    
    with open(train_path, 'r') as f:
        train_data = json.load(f)
        train_image_len = len(train_data['images'])
    with open(test_path, 'r') as f:
        test_data = json.load(f)
        test_image_len = len(test_data['images'])
    with open(val_path, 'r') as f:
        val_data = json.load(f)
        val_image_len = len(val_data['images'])
        
    total_image_len = train_image_len + test_image_len + val_image_len
    print("="*40)
    print(f"Dataset: {dataset}")
    print(f"Train ratio: {train_image_len/total_image_len:.2%} ({train_image_len})")
    print(f"Test ratio: {test_image_len/total_image_len:.2%} ({test_image_len})")
    print(f"Val ratio: {val_image_len/total_image_len:.2%} ({val_image_len})")
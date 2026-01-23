import tarfile
from pathlib import Path
from tqdm import tqdm

def get_dataset_image_counts(wds_root, split="train"):
    """
    WDS 루트 디렉토리를 순회하며 데이터셋별 실제 이미지(샘플) 개수를 산출합니다.
    """
    wds_root = Path(wds_root)
    dataset_dirs = sorted(list(wds_root.glob(f"*_{split}")))
    
    dataset_counts = {}
    print(f"\n🔍 [지적 분석] {split} 분할의 이미지 개수를 집계합니다...")
    
    for d_dir in dataset_dirs:
        d_name = d_dir.name.replace(f"_{split}", "")
        shards = sorted(list(d_dir.glob("shard-*.tar")))
        
        sample_count = 0
        for shard_path in tqdm(shards, desc=f"Counting {d_name}", leave=False):
            with tarfile.open(shard_path, 'r') as tar:
                # WDS에서 한 샘플은 동일한 키를 가진 파일들의 집합입니다.
                # 보통 .pth 파일 하나당 샘플 하나이므로 키의 중복을 제거하여 셉니다.
                keys = {member.name.split('.')[0] for member in tar.getmembers() if member.isfile()}
                sample_count += len(keys)
        
        dataset_counts[d_name] = sample_count
        
    return dataset_counts

if __name__ == "__main__":
    wds_root = Path("/home/vmg/Desktop/layout2video/datasets/betr_wds")
    train_counts = get_dataset_image_counts(wds_root, split="train")
    val_counts = get_dataset_image_counts(wds_root, split="val")

    print("\n📊 Train dataset image counts:")
    for dataset, count in train_counts.items():
        print(f"  {dataset}: {count}")

    print("\n📊 Validation dataset image counts:")
    for dataset, count in val_counts.items():
        print(f"  {dataset}: {count}")
import torch
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import sys

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src_betr"))

from data.image_dataloader import build_image_dataloader

class WelfordStats:
    def __init__(self, shape):
        self.n = 0
        self.mean = torch.zeros(shape)
        self.M2 = torch.zeros(shape)

    def update(self, x):
        batch_size = x.size(0)
        for i in range(batch_size):
            sample = x[i].cpu()
            self.n += 1
            delta = sample - self.mean
            self.mean += delta / self.n
            delta2 = sample - self.mean
            self.M2 += delta * delta2

    @property
    def finalized_mean(self):
        return self.mean

    @property
    def finalized_std(self):
        if self.n < 2:
            return torch.zeros_like(self.mean)
        return torch.sqrt(self.M2 / self.n)

def main():
    # 1. 설정 로드
    config_path = PROJECT_ROOT / "src_betr" / "configs" / "betr_config.yaml"
    cfg = OmegaConf.load(config_path)
    
    # 2. 데이터로더 설정 (학습셋 전체 확인을 위해 split="train" 사용)
    # 현재 overfitting_test를 위해 image_dataloader를 사용하신다고 했으므로 이를 활용합니다.
    dataloader = build_image_dataloader(
        root_dir=Path(cfg.json_root),
        data_dir=Path(cfg.data_root),
        seed=cfg.seed,
        split="train", # 실제 학습 데이터 분포 확인
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        filter=True # 실제 학습 시와 동일한 필터링 조건 적용
    )

    # 3. 통계 계산기 초기화 (image_dataloader의 타겟 형상 기준)
    stats = {
        "corners": WelfordStats((8, 2)),            # gt_corners [B, 8, 2]
        "depths": WelfordStats((8, 1)),       # gt_depths [B, 8, 1]
    }

    print(f"📊 Starting statistics computation for {len(dataloader.dataset)} samples...")

    # 4. 데이터 순회
    for batch in tqdm(dataloader, desc="Computing Stats"):
        stats["corners"].update(batch["gt_corners"])
        stats["depths"].update(batch["gt_depths"])
        # scalar 값은 차원을 맞춰서 전달\

    # 5. 결과 출력
    print("\n" + "="*50)
    print("📈 Training Set Target Statistics (Canonical Depth)")
    print("="*50)
    
    for name, s in stats.items():
        mean = s.finalized_mean
        std = s.finalized_std
        print(f"\n[ {name.upper()} ]")
        if mean.dim() == 0 or (mean.dim() == 1 and mean.size(0) == 1):
            print(f"  Mean: {mean.item():.6f}")
            print(f"  Std : {std.item():.6f}")
        else:
            # 다차원인 경우 전체 평균과 대표값을 출력
            print(f"  Mean (Overall): {mean.mean().item():.6f}")
            print(f"  Std (Overall) : {std.mean().item():.6f}")
            print(f"  Raw Mean: {mean.tolist()}")
            print(f"  Raw Std : {std.tolist()}")
    
    print("="*50)
    print("\n💡 이 값을 데이터로더에서 (target - mean) / std 로 적용하여 정규화하세요.")

if __name__ == "__main__":
    main()
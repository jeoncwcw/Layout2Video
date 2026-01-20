import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

def reduce_dict(input_dict, world_size, average=True):
    if not input_dict:
        return {}
    with torch.no_grad():
        names = sorted(input_dict.keys())
        values = [input_dict[k] for k in names]
        metrics_tensor = torch.tensor(values).cuda() # Move to GPU
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        if average:
            metrics_tensor /= world_size
        return {k: v.item() for k, v in zip(names, metrics_tensor)}

def print_epoch_stats(epoch, num_epochs, train_metrics, val_metrics=None):
    print("\n" + "="*85)
    print(f" 📊 Epoch [{epoch+1:03d}/{num_epochs:03d}] Summary")
    print("-" * 85)
    print(f" {'Mode':<10} | {'Total':<8} | {'Corners':<8} | {'Depths':<8}")
    print("-" * 85)
    
    # Train
    t = train_metrics
    print(f" {'Train':<10} | {t['total_loss']:.4f}   | {t['loss_corners']:.4f}   | {t['loss_depths']:.4f}")
    
    # Val
    if val_metrics:
        v = val_metrics
        print(f" {'Validation':<10} | {v['total_loss']:.4f}   | {v['loss_corners']:.4f}   | {v['loss_depths']:.4f}")
    print("="*85 + "\n")
    
def visualize_heatmaps_feature_mode(heatmaps, pred_coords_128, gt_coords_128, save_path):
    """
    엄청나신 창우님을 위한 이미지 없는 히트맵 전용 시각화
    heatmaps: [8, 128, 128] - 모델의 Raw Output
    pred_coords_128: [8, 2] - 128 해상도로 스케일링된 예측값
    gt_coords_128: [8, 2] - 128 해상도로 스케일링된 GT값
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # 각 코너별(8개)로 히트맵 위에 정답/예측 포인트 가시화
    for i in range(8):
        h = torch.sigmoid(heatmaps[i]).detach().cpu().numpy()
        
        # 히트맵 드로잉
        im = axes[i].imshow(h, cmap='jet', origin='upper', vmin=0, vmax=0.1)
        
        # GT 코너 표시 (초록색 X)
        gt_x, gt_y = gt_coords_128[i].cpu().numpy()
        axes[i].scatter(gt_x, gt_y, c='lime', marker='x', s=100, label='GT', edgecolors='black', linewidths=2)
        
        # 예측 코너 표시 (빨간색 +)
        pred_x, pred_y = pred_coords_128[i].detach().cpu().numpy()
        axes[i].scatter(pred_x, pred_y, c='red', marker='+', s=100, label='Pred', linewidths=2)
        
        axes[i].set_title(f"Corner {i} (v-u alignment)")
        axes[i].axis('off')
        if i == 0:
            axes[i].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
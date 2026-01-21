import os
import random
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from collections import defaultdict
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

from models.betr import BETRModel
from losses.criterion import BETRLoss
from data.wds_dataloader import build_wds_feature_dataloader
from utils import set_seed, reduce_dict, print_epoch_stats, visualize_heatmaps

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    mp.set_sharing_strategy('file_system')
    torch.cuda.set_device(rank)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=rank)
    except:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    

def train_worker(rank, world_size, cfg):
    setup(rank, world_size)
    set_seed(cfg.get("seed", 42), rank)
    device = torch.device(f"cuda:{rank}")
    
    model = BETRModel(cfg).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    gt_size = cfg.data.dino_image_size
    criterion = BETRLoss(
        lambda_fine=cfg.loss_weights.lambda_,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    
    total_data_kwargs = {
        "wds_root": Path(cfg.wds_root),
        "batch_size": cfg.batch_size,
        "dino_img_size": gt_size,
        "num_workers": cfg.num_workers,
        "world_size": world_size
    }
    train_dataloader = build_wds_feature_dataloader(**total_data_kwargs, split="train", epoch_length=cfg.epoch_length)
    val_dataloader = build_wds_feature_dataloader(**total_data_kwargs, split="val", epoch_length=cfg.epoch_length//10)
    
    num_epochs = cfg.num_epochs
    checkpoint_dir = SRC_ROOT / "checkpoints" / cfg.model_name
    vis_dir = PROJECT_ROOT / "test" / cfg.model_name / "heatmap_vis"
    best_val_loss = float('inf')
    if rank == 0:   
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 50)
        print(f"Starting training for {num_epochs} epochs...")
        print("=" * 50)
    scaler = torch.amp.GradScaler('cuda')
    num_batches_per_epoch = cfg.epoch_length // (cfg.batch_size * world_size)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_meter = defaultdict(float)
        train_num_samples = 0
        iterator = train_dataloader
        if rank == 0:
            iterator = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=True, total=num_batches_per_epoch)
        for i, batch in enumerate(iterator):
            if i % 500 == 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(
                    bbx2d_tight=batch_gpu["2d_bbx"],
                    mask=batch_gpu["padding_mask"],
                    f_metric=batch_gpu["feat_metric"],
                    f_mono=batch_gpu["feat_mono"],
                    f_dino=batch_gpu["feat_dino"],
                )
                loss_dict = criterion(outputs, batch_gpu)
                total_loss = loss_dict["total_loss"]
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for k, v in loss_dict.items():
                    train_meter[k] += v.item() * batch_gpu["2d_bbx"].size(0)
                train_num_samples += batch_gpu["2d_bbx"].size(0)
            if rank == 0:
                iterator.set_postfix(loss=f"{total_loss.item():.4f}")
        
        # Train metric
        local_avg_metrics = {k: v / train_num_samples for k, v in train_meter.items()}
        train_avg_metrics = reduce_dict(local_avg_metrics, world_size, average=True)
        
        # Validation phase
        global_val_metrics = None
        if (epoch+1) % cfg.get("val_interval", 1) == 0:
            model.eval()
            val_meter = defaultdict(float)
            val_num_samples = 0
            if rank == 0:
                print(f"Starting validation for Epoch {epoch+1}...")
            
            with torch.inference_mode():
                for i, val_batch in enumerate(val_dataloader):
                    val_batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
                    val_outputs = model(
                        bbx2d_tight=val_batch_gpu["2d_bbx"],
                        mask=val_batch_gpu["padding_mask"],
                        f_metric=val_batch_gpu["feat_metric"],
                        f_mono=val_batch_gpu["feat_mono"],
                        f_dino=val_batch_gpu["feat_dino"],
                    )
                    val_loss_dict = criterion(val_outputs, val_batch_gpu)
                    for k, v in val_loss_dict.items():
                        val_meter[k] += v.item() * val_batch_gpu["2d_bbx"].size(0)    
                    val_num_samples += val_batch_gpu["2d_bbx"].size(0)
                    if rank ==0 and i == 0:
                        h_maps = val_outputs['corner heatmaps'][0]  # [8, 128, 128]
                        p_coords_128 = val_outputs['corner coords'][0] / 4.0
                        g_coords_128 = val_batch['gt_corners'][0] * 128.0
                        save_name = vis_dir / f"epoch_{epoch+1:03d}.png"
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        visualize_heatmaps(h_maps, p_coords_128, g_coords_128, save_name)
                        print(f"Saved heatmap visualization to {save_name}")
            try:
                dist.barrier(device_ids=[rank])
            except:
                dist.barrier()
            
            total_samples_tensor = torch.tensor([val_num_samples], device=device)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            total_val_samples = total_samples_tensor.item()
            
            if total_val_samples > 0:
                local_sum_metrics = {k: v for k, v in val_meter.items()}
                global_sum_metrics = reduce_dict(local_sum_metrics, world_size, average=False)
                global_val_metrics = {k: v / total_val_samples for k, v in global_sum_metrics.items()}
            else:
                if rank == 0:
                    print("⚠️ No validation samples found!")
                
        if rank == 0:
            print_epoch_stats(epoch, num_epochs, train_avg_metrics, global_val_metrics)
            if global_val_metrics:
                current_val_loss = global_val_metrics["total_loss"]
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    checkpoint_path = checkpoint_dir / f"best.pth"
                    torch.save(model.module.state_dict(), checkpoint_path)
                    print(f"Saved best model with val loss {best_val_loss:.4f} at {checkpoint_path}")
            if (epoch + 1) % cfg.get("save_interval", 5) == 0:
                checkpoint_path = checkpoint_dir / f"epoch{epoch+1:03d}.pth"
                torch.save(model.module.state_dict(), checkpoint_path)
                print(f"Saved checkpoint at {checkpoint_path}")  
    cleanup()
    
def main():
    cfg_path = SRC_ROOT / "configs" / "betr_config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.feature_mode = True
    world_size = torch.cuda.device_count()
    if (SRC_ROOT / "checkpoints" / cfg.model_name).exists():
        print("Model already trained. Exiting.")
        return
    mp.spawn(train_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    
if __name__ == "__main__":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    mp.set_sharing_strategy('file_system')
    main()
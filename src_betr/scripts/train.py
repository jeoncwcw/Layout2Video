import os
import random
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from omegaconf import OmegaConf
import sys
from timm.utils import ModelEmaV2

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

from models.betr import BETRModel
from losses.criterion import BETRLoss
from data.wds_dataloader import build_wds_feature_dataloader
from utils import set_seed, print_epoch_stats, train_one_epoch, evaluate, get_scheduler, CornerGeometryMetric, get_parameter_groups

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    # mp.set_sharing_strategy('file_system')
    torch.cuda.set_device(rank)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=rank)
    except:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cudnn.rnn.fp32_precision = "tf32"
    torch.set_float32_matmul_precision('high')

def cleanup():
    dist.destroy_process_group()
    

def train_worker(rank, world_size, cfg):
    # Base settings
    try:
        setup(rank, world_size)
        set_seed(cfg.get("seed", 42), rank)
        device = torch.device(f"cuda:{rank}")
        
        model = BETRModel(cfg).to(device)
        model = torch.compile(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False, gradient_as_bucket_view=True)
        model_ema = ModelEmaV2(model, decay=0.999, device=device)
        gt_size = cfg.data.dino_image_size
        checkpoint_dir = SRC_ROOT / "checkpoints" / cfg.model_name
        vis_dir = PROJECT_ROOT / "test" / cfg.model_name / "heatmap_vis"
        num_epochs = cfg.num_epochs
        num_batches_per_epoch = cfg.epoch_length // (cfg.batch_size * world_size)
        
        # Learning components
        criterion = BETRLoss(
            start_lambda_fine=cfg.loss_weights.start_lambda_,
            end_lambda_fine=cfg.loss_weights.end_lambda_,
            loss_depth=cfg.loss_weights.depth,
            start_peak = cfg.loss_weights.start_peak,
            end_peak = cfg.loss_weights.end_peak,
            total_epochs = num_epochs,
        ).to(device)
        optimizer = optim.AdamW(
            get_parameter_groups(model, weight_decay=cfg.weight_decay),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        scheduler = get_scheduler(optimizer, cfg, num_batches_per_epoch)
        
        total_data_kwargs = {
            "cfg": cfg,
            "wds_root": Path(cfg.wds_root),
            "batch_size": cfg.batch_size,
            "dino_img_size": gt_size,
            "num_workers": cfg.num_workers,
            "world_size": world_size
        }
        train_dataloader, aug_obj = build_wds_feature_dataloader(**total_data_kwargs, split="train", epoch_length=cfg.epoch_length)
        val_dataloader, _ = build_wds_feature_dataloader(**total_data_kwargs, split="val", epoch_length=cfg.epoch_length//10)
        
        best_dist_metric = float('inf')
        if rank == 0:   
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print("=" * 50)
            print(f"Starting training for {num_epochs} epochs...")
            print("=" * 50)
        scaler = torch.amp.GradScaler('cuda')
        val_metric = CornerGeometryMetric(device=device)
        ema_metric = CornerGeometryMetric(device=device)
        
        for epoch in range(num_epochs):
            if aug_obj is not None:
                aug_obj.set_epoch(epoch)
            # Training phase
            train_metrics = train_one_epoch(
                model=model, model_ema=model_ema,
                train_dataloader=train_dataloader,
                rank=rank, epoch=epoch, num_batches_per_epoch=num_batches_per_epoch,
                device=device, criterion=criterion,
                grad_accum_steps=cfg.get("grad_accum_steps", 1),
                scaler=scaler, optimizer=optimizer, scheduler=scheduler,
                world_size=world_size,
            )
            
            # Validation phase
            global_val_loss, global_ema_loss, global_val_dists, global_ema_dists = (None, None, None, None)
            if (epoch+1) % cfg.get("val_interval", 1) == 0:
                global_val_loss, global_ema_loss, global_val_dists, global_ema_dists = evaluate(
                    model=model, model_ema=model_ema,
                    val_metric=val_metric, ema_metric=ema_metric,
                    rank=rank,
                    epoch=epoch,
                    val_dataloader=val_dataloader,
                    device=device,
                    criterion=criterion,
                    world_size=world_size,
                    vis_dir=vis_dir,
                )
                
            # Logging and checkpointing
            if rank == 0:
                print_epoch_stats(epoch, num_epochs, train_metrics, global_val_loss, global_ema_loss)
                if global_val_loss:
                    print("---- Validation Corner Geometry Metrics ----")
                    print(f"Standard Model - Average UV Error: {global_val_dists['avg_uv_error']:.4f} px, Average Depth Error: {global_val_dists['avg_depth_error']:.4f} meters")
                    print(f"EMA Model      - Average UV Error: {global_ema_dists['avg_uv_error']:.4f} px, Average Depth Error: {global_ema_dists['avg_depth_error']:.4f} meters")
                    print("--------------------------------------------")
                    # Checkpointing based on metric
                    min_score = min(global_val_dists['mixed_score'], global_ema_dists['mixed_score'])
                    if min_score < best_dist_metric:
                        best_dist_metric = min_score
                        checkpoint_path = checkpoint_dir / f"best_dist.pth"
                        if global_ema_dists['mixed_score'] < global_val_dists['mixed_score']:
                            torch.save(model_ema.module.state_dict(), checkpoint_path)
                        else:
                            torch.save(model.module.state_dict(), checkpoint_path)
                        print(f"Saved best distance metric model with mixed error {min_score:.4f} at {checkpoint_path}")
                # Regular checkpointing
                if (epoch + 1) % cfg.get("save_interval", 5) == 0:
                    checkpoint_path = checkpoint_dir / f"epoch{epoch+1:03d}.pth"
                    torch.save(model.module.state_dict(), checkpoint_path)
                    print(f"Saved checkpoint at {checkpoint_path}")  
    except Exception as e:
        print(f"Exception in rank {rank}: {e}")
    finally:
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
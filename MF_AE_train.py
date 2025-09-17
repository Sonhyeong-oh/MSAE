from MF_AE_Model import MultiFilterMemAE  # 새 모델 import
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image
import os
from tqdm import tqdm

torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)

os.add_dll_directory(r"C:\Users\Admin\anaconda3\bin")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class SingleFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = [os.path.join(root, f) for f in os.listdir(root)
                      if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def main():
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # 새로운 MultiFilterMemAE 모델
    model = MultiFilterMemAE(
        feature_dim=256,        # 피처 차원
        mem_dim=512,           # 메모리 슬롯 개수
        patch_size=7,          # 패치 크기
        shrink_thres=0.0025,   # MemAE shrinkage threshold
        use_gabor=False        # Gabor 필터 사용 여부 (시작은 False)
    ).cuda()
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-4,           # MultiFilter는 더 작은 학습률로 시작
        weight_decay=0.01,
        fused=True
    )

    # 학습률 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15, eta_min=1e-6)

    # 데이터 전처리 (정규화 제거 - MultiFilter는 원본 이미지 [0,1] 범위 사용)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        # 정규화 없음 - 필터링을 위해 원본 범위 유지
    ])

    # 데이터셋
    train_dir = r"C:/Users/Admin/Desktop/alaska2-image-steganalysis/Cover/train"
    val_dir = r"C:/Users/Admin/Desktop/alaska2-image-steganalysis/Cover/val"

    train_dataset = SingleFolderDataset(train_dir, transform=transform)
    val_dataset   = SingleFolderDataset(val_dir,   transform=transform)

    # DataLoader (배치 크기 줄임 - 더 많은 브랜치로 인한 메모리 사용량 증가)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # 24->16으로 줄임
        shuffle=True,
        num_workers=6,
        pin_memory=True, 
        persistent_workers=True, 
        pin_memory_device="cuda",
        prefetch_factor=2,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True, 
        persistent_workers=True, 
        pin_memory_device="cuda",
        prefetch_factor=2
    )

    # 체크포인트 설정
    SAVE_DIR = "C:/Users/Admin/Desktop/Python/StegDetector/MultiFilter_AE/checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_path = os.path.join(SAVE_DIR, "multifilter_memae_best.pth")
    last_path = os.path.join(SAVE_DIR, "multifilter_memae_last.pth")

    patience = 7
    min_delta = 1e-4
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0

    accum = 4  # gradient accumulation
    epochs = 25

    # Warmup 설정
    warmup_steps = len(train_loader) * 2
    
    print("Starting Multi-Filter MemAE training...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Batch size: 16, Accumulation steps: {accum}")
    print(f"Using Gabor filters: {model.use_gabor}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        train_logs_sum = {
            'mse_loss': 0.0, 
            'ssim_loss': 0.0,
            'dct_loss': 0.0,
            'entropy': 0.0, 
            'total_loss': 0.0
        }
        steps = 0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False)

        opt.zero_grad(set_to_none=True)

        for step, imgs in enumerate(pbar):
            imgs = imgs.cuda(non_blocking=True)
            
            # Warmup learning rate
            global_step = epoch * len(train_loader) + step
            if global_step < warmup_steps:
                lr_scale = global_step / warmup_steps
                for param_group in opt.param_groups:
                    param_group['lr'] = 2e-4 * lr_scale
            
            # Mixed Precision Training
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, logs = model(imgs, compute_loss=True)
            
            # NaN 체크
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at step {step}, skipping...")
                continue
            
            # Scaled backward pass
            scaler.scale(loss / accum).backward()
            
            # 로그 누적
            for k in train_logs_sum:
                if k in logs:
                    train_logs_sum[k] += logs[k]

            # Gradient accumulation
            if (step + 1) % accum == 0:
                # Gradient clipping
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item()
            steps += 1
            
            # 진행 상황 표시
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{opt.param_groups[0]['lr']:.2e}"
            })

        # 남은 누적 스텝 처리
        if (step + 1) % accum != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        avg_tr = epoch_loss / max(1, steps)
        avg_tr_logs = {k: v / max(1, steps) for k, v in train_logs_sum.items()}

        # Validation
        model.eval()
        val_loss = 0.0
        val_logs_sum = {
            'mse_loss': 0.0, 
            'ssim_loss': 0.0,
            'dct_loss': 0.0,
            'entropy': 0.0, 
            'total_loss': 0.0
        }
        vsteps = 0
        
        with torch.no_grad():
            pbar_v = tqdm(val_loader, desc=f"Val  {epoch+1}/{epochs}", leave=False)
            for imgs in pbar_v:
                imgs = imgs.cuda(non_blocking=True)
                
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, logs = model(imgs, compute_loss=True)
                
                # NaN 체크
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    for k in val_logs_sum:
                        if k in logs:
                            val_logs_sum[k] += logs[k]
                    vsteps += 1
                
                pbar_v.set_postfix(vloss=f"{loss.item():.4f}")

        avg_val = val_loss / max(1, vsteps) if vsteps > 0 else float('inf')
        avg_val_logs = {k: v / max(1, vsteps) for k, v in val_logs_sum.items()} if vsteps > 0 else {}

        # Learning rate scheduler step
        scheduler.step()
        
        print(f"[Epoch {epoch+1}] train {avg_tr:.4f} | val {avg_val:.4f} | lr {opt.param_groups[0]['lr']:.2e}")
        print(f" train_losses: mse={avg_tr_logs.get('mse_loss', 0):.4f}, ssim={avg_tr_logs.get('ssim_loss', 0):.4f}, dct={avg_tr_logs.get('dct_loss', 0):.4f}, entropy={avg_tr_logs.get('entropy', 0):.4f}, total={avg_tr_logs.get('total_loss', 0):.4f}")
        if vsteps > 0:
            print(f" val_losses: mse={avg_val_logs.get('mse_loss', 0):.4f}, ssim={avg_val_logs.get('ssim_loss', 0):.4f}, dct={avg_val_logs.get('dct_loss', 0):.4f}, entropy={avg_val_logs.get('entropy', 0):.4f}, total={avg_val_logs.get('total_loss', 0):.4f}")

        # 브랜치 가중치 출력 (마지막 배치에서)
        if 'branch_weights' in locals():
            try:
                # 모델에서 브랜치 가중치 가져오기
                with torch.no_grad():
                    sample_output = model(imgs[:2], return_details=True)
                    if 'branch_weights' in sample_output:
                        weights = sample_output['branch_weights'].cpu().numpy()
                        branch_names = ['Original', 'LPF', 'LoG', 'HPF']
                        if model.use_gabor:
                            branch_names.append('Gabor')
                        weight_str = ', '.join([f"{name}={w:.3f}" for name, w in zip(branch_names, weights)])
                        print(f" branch_weights: {weight_str}")
            except:
                pass

        # Checkpoint / Early Stop
        if avg_val < best_val - min_delta:
            best_val = avg_val
            best_epoch = epoch + 1
            no_improve = 0
            
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": best_val,
                "train_loss": avg_tr,
                "use_gabor": model.use_gabor
            }, best_path)
            
            print(f"Improved! Saved best checkpoint (val={best_val:.6f})")
        else:
            no_improve += 1
            print(f"No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            print(f"   Best epoch: {best_epoch}, Best val: {best_val:.6f}")
            break

        # 메모리 정리
        if (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()

    # Save last checkpoint
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "use_gabor": model.use_gabor
    }, last_path)
    print(f"Saved last checkpoint to: {last_path}")

    # Example outputs
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for imgs in train_loader:
            imgs = imgs.cuda(non_blocking=True)
            
            # 기본 출력
            reconstructed = model(imgs[:2])
            print("Output shapes:")
            print(f"  Input: {imgs[:2].shape}")
            print(f"  Reconstructed: {reconstructed.shape}")
            
            # 세부 정보 출력
            detailed_output = model(imgs[:2], return_details=True)
            print("Detailed output keys:")
            for key, value in detailed_output.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {type(value)}")
            break

if __name__ == "__main__":
    main()
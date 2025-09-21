from MFwM_AE import MultiFilterAutoEncoder  # 새 모델 import
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    
    # 새로운 MultiFilterAutoEncoder 모델 (수정된 파라미터)
    model = MultiFilterAutoEncoder(
        feature_dim=256        # 피처 차원만 필요
    ).cuda()
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # 옵티마이저 정의
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 스케줄러 정의: patience를 15로 설정하여 15 에포크 동안 개선이 없으면 학습률을 0.1배로 줄임
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 정규화 없음 - 필터링을 위해 원본 범위 [0,1] 유지
    ])

    # 데이터셋
    train_dir = r"C:/Users/Admin/Desktop/alaska2-image-steganalysis/Cover/train"
    val_dir = r"C:/Users/Admin/Desktop/alaska2-image-steganalysis/Cover/val"

    train_dataset = SingleFolderDataset(train_dir, transform=transform)
    val_dataset   = SingleFolderDataset(val_dir,   transform=transform)

    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  
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
        batch_size=8,
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
    best_path = os.path.join(SAVE_DIR, "multifilter_ae_best.pth")
    last_path = os.path.join(SAVE_DIR, "multifilter_ae_last.pth")

    patience = 10
    min_delta = 1e-4
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0

    accum = 4  # gradient accumulation
    epochs = 100

    # Warmup 설정
    warmup_steps = len(train_loader) * 2
    
    print("Starting Multi-Filter AutoEncoder training...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Batch size: 16, Accumulation steps: {accum}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        train_logs_sum = {
            'base_mse': 0.0,
            'hybrid': 0.0,
            'dct': 0.0,
            'original_spec': 0.0,
            'hybrid_spec': 0.0,
            'dct_spec': 0.0,
            'total': 0.0
        }
        steps = 0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False)

        optimizer.zero_grad(set_to_none=True)

        for step, imgs in enumerate(pbar):
            imgs = imgs.cuda(non_blocking=True)
            
            # Warmup learning rate
            global_step = epoch * len(train_loader) + step
            if global_step < warmup_steps:
                lr_scale = global_step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2e-4 * lr_scale
            
            # Mixed Precision Training
            with torch.autocast('cuda', dtype=torch.bfloat16):
                # include_specialization_loss=True로 브랜치 특화 손실 포함
                loss, logs = model(imgs, compute_loss=True, include_specialization_loss=True)
            
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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item()
            steps += 1
            
            # 진행 상황 표시
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        # 남은 누적 스텝 처리
        if (step + 1) % accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_tr = epoch_loss / max(1, steps)
        avg_tr_logs = {k: v / max(1, steps) for k, v in train_logs_sum.items()}

        # Validation
        model.eval()
        val_loss = 0.0
        val_logs_sum = {
            'base_mse': 0.0,
            'hybrid': 0.0,
            'dct': 0.0,
            'original_spec': 0.0,
            'hybrid_spec': 0.0,
            'dct_spec': 0.0,
            'total': 0.0
        }
        vsteps = 0
        
        with torch.no_grad():
            pbar_v = tqdm(val_loader, desc=f"Val  {epoch+1}/{epochs}", leave=False)
            for imgs in pbar_v:
                imgs = imgs.cuda(non_blocking=True)
                
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, logs = model(imgs, compute_loss=True, include_specialization_loss=True)
                
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
        scheduler.step(val_loss)
        
        print(f"[Epoch {epoch+1}] train {avg_tr:.4f} | val {avg_val:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")
        print(f" train_losses: base_mse={avg_tr_logs.get('base_mse', 0):.4f}, hybrid={avg_tr_logs.get('hybrid', 0):.4f}, dct={avg_tr_logs.get('dct', 0):.4f}, total={avg_tr_logs.get('total', 0):.4f}")
        if vsteps > 0:
            print(f" val_losses: base_mse={avg_val_logs.get('base_mse', 0):.4f}, hybrid={avg_val_logs.get('hybrid', 0):.4f}, dct={avg_val_logs.get('dct', 0):.4f}, total={avg_val_logs.get('total', 0):.4f}")
            
        # 브랜치 특화 손실도 출력
        if vsteps > 0:
            print(f" specialization: orig={avg_tr_logs.get('original_spec', 0):.4f}, hybrid={avg_tr_logs.get('hybrid_spec', 0):.4f}, dct={avg_tr_logs.get('dct_spec', 0):.4f}")

        # 어텐션 맵 분석 (샘플링)
        if epoch % 5 == 0:  # 5 에포크마다
            try:
                with torch.no_grad():
                    sample_details = model(imgs[:2], return_dcts=True)
                    if 'attention_maps' in sample_details:
                        att_maps = sample_details['attention_maps']
                        if 'unified_attention' in att_maps:
                            unified_att = att_maps['unified_attention']
                            att_mean = unified_att.mean().item()
                            att_std = unified_att.std().item()
                            print(f" attention_stats: mean={att_mean:.4f}, std={att_std:.4f}")
            except Exception as e:
                print(f" attention analysis failed: {e}")

        # Checkpoint / Early Stop
        if avg_val < best_val - min_delta:
            best_val = avg_val
            best_epoch = epoch + 1
            no_improve = 0
            
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": best_val,
                "train_loss": avg_tr,
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
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, last_path)
    print(f"Saved last checkpoint to: {last_path}")

    # Example outputs
    print("\n=== Model Output Analysis ===")
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
            detailed_output = model(imgs[:2], return_dcts=True)
            print("\nDetailed output keys:")
            for key, value in detailed_output.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} keys")
                    for subkey, subvalue in value.items():
                        if hasattr(subvalue, 'shape'):
                            print(f"    {subkey}: {subvalue.shape}")
                        else:
                            print(f"    {subkey}: {type(subvalue)}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {len(value)} items")
                    for i, item in enumerate(value):
                        if hasattr(item, 'shape'):
                            print(f"    [{i}]: {item.shape}")
                else:
                    print(f"  {key}: {type(value)}")
            
            # 손실 분석
            loss, logs = model(imgs[:2], compute_loss=True, include_specialization_loss=True)
            print(f"\nSample Loss Analysis:")
            for key, value in logs.items():
                print(f"  {key}: {value:.6f}")
            
            break

if __name__ == "__main__":
    main()
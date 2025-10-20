# -*- coding: utf-8 -*-
"""
MSCAE Fine-Tuning Script for the full StegaNetRGB model.

This script loads a pretrained encoder/decoder and fine-tunes the entire
end-to-end model for classification.
"""

import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler

# Import the full RGB classification model
from MSCAE_RGBmodel import StegaNetRGB

# --- Dataset for loading images directly ---
class ImageDataset(Dataset):
    def __init__(self, paths: list, labels: list):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path, label = self.paths[idx], self.labels[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            # CRITICAL FIX: NO normalization - keep [0, 255] range to match SRM filter expectations
            # SRM filters are designed for pixel differences in [0, 255] range
            img_np = np.array(img, dtype=np.float32)
            img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1)))

            y = torch.tensor(label, dtype=torch.long)
            return img_tensor, y
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}, Path: {img_path}")
            # Return a dummy tensor and label
            return torch.zeros((3, 256, 256)), torch.tensor(0, dtype=torch.long)

# --- Training & Evaluation Functions ---
def run_epoch(model, loader, optimizer, device, scaler: GradScaler, use_amp: bool, accumulation_steps: int):
    """
    Training epoch with enhanced monitoring.

    Returns:
        avg_loss: Average classification loss
        accuracy: Classification accuracy
        recon_ratio: Average ratio of stego/cover reconstruction errors (for monitoring)
    """
    model.train()
    total_loss, total_correct = 0.0, 0
    recon_ratios = []

    pbar = tqdm(loader, desc="Training", leave=False)

    optimizer.zero_grad()
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, reconstructed = model(images)

            # Classification Loss ONLY
            # Pretrain already learned differential reconstruction via SNR Loss
            # No need for reconstruction loss during fine-tuning
            cls_loss = F.cross_entropy(logits, labels)

            # Use only classification loss
            loss = cls_loss

            # Normalize loss to account for accumulation
            loss = loss / accumulation_steps

        # Store loss values for logging
        cls_loss_val = cls_loss.item()

        # Accumulate scaled gradients
        scaler.scale(loss).backward()

        # Perform an optimizer step every `accumulation_steps`
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Monitor reconstruction error ratio (for observing model behavior)
        with torch.no_grad():
            # Multi-branch SRM filtering for monitoring
            filtered_branches = [srm_branch(images) for srm_branch in model.srm_branches]
            srm_filtered = torch.cat(filtered_branches, dim=1)  # (B, 90, H, W)
            recon_error = torch.abs(srm_filtered - reconstructed).mean(dim=[1, 2, 3])

            cover_mask = (labels == 0)
            stego_mask = (labels == 1)

            if cover_mask.any() and stego_mask.any():
                cover_err = recon_error[cover_mask].mean().item()
                stego_err = recon_error[stego_mask].mean().item()
                if cover_err > 0:
                    recon_ratios.append(stego_err / cover_err)

        # Multiply by accumulation_steps to get the actual loss for logging
        total_loss += (loss.item() * accumulation_steps) * images.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == labels).sum().item()

        # Enhanced progress bar with reconstruction ratio
        postfix = {
            "cls": f"{cls_loss_val:.4f}",
            "acc": f"{(preds == labels).float().mean():.4f}"
        }
        if recon_ratios:
            postfix["rec_ratio"] = f"{np.mean(recon_ratios[-10:]):.2f}"  # Last 10 batches
        pbar.set_postfix(postfix)

    avg_recon_ratio = np.mean(recon_ratios) if recon_ratios else 1.0
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset), avg_recon_ratio

@torch.no_grad()
def evaluate(model, loader, device):
    """
    Enhanced evaluation with reconstruction error monitoring.

    Returns:
        total_loss: Average classification loss
        accuracy: Classification accuracy
        auc: ROC-AUC score
        recon_stats: Dict with reconstruction error statistics
    """
    model.eval()
    total_loss, total_correct = 0.0, 0
    probs_all, labels_all = [], []

    # Track reconstruction errors separately for cover and stego
    recon_errors_cover = []
    recon_errors_stego = []

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, reconstructed = model(images)
            loss = F.cross_entropy(logits, labels)

            # Calculate reconstruction error for monitoring
            # Note: StegaNetRGB's decoder outputs 90-channel SRM-filtered reconstruction
            # We compute the mean absolute reconstruction error per sample
            with torch.no_grad():
                # Get SRM-filtered input for fair comparison (multi-branch)
                filtered_branches = [srm_branch(images) for srm_branch in model.srm_branches]
                srm_filtered = torch.cat(filtered_branches, dim=1)  # (B, 90, H, W)
                recon_error = torch.abs(srm_filtered - reconstructed).mean(dim=[1, 2, 3])  # (B,)

                # Separate by label (0=cover, 1=stego)
                cover_mask = (labels == 0)
                stego_mask = (labels == 1)

                if cover_mask.any():
                    recon_errors_cover.extend(recon_error[cover_mask].cpu().numpy().tolist())
                if stego_mask.any():
                    recon_errors_stego.extend(recon_error[stego_mask].cpu().numpy().tolist())

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        probs = F.softmax(logits, dim=1)[:, 1]
        probs_all.append(probs.cpu().numpy())
        labels_all.append(labels.cpu().numpy())

    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    auc = roc_auc_score(labels_all, probs_all)

    # Calculate reconstruction error statistics
    recon_stats = {
        'cover_mean': np.mean(recon_errors_cover) if recon_errors_cover else 0.0,
        'stego_mean': np.mean(recon_errors_stego) if recon_errors_stego else 0.0,
        'cover_std': np.std(recon_errors_cover) if recon_errors_cover else 0.0,
        'stego_std': np.std(recon_errors_stego) if recon_errors_stego else 0.0,
        'ratio': (np.mean(recon_errors_stego) / np.mean(recon_errors_cover)) if recon_errors_cover and recon_errors_stego else 0.0
    }

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset), auc, recon_stats

def main():
    parser = argparse.ArgumentParser(description="MSCAE StegaNetRGB Fine-Tuning with Balanced Multi-Algorithm Dataset")
    # Paths
    parser.add_argument("--pretrained_path", type=str, default="./checkpoints_cae/cae_rgb_balanced_stage1.pth",
                        help="Path to pretrained CAE model for weight initialization.")
    parser.add_argument("--cover_dir", type=str, default=str(Path.home() / "ALASKA2" / "Cover_small"),
                        help="Path to cover images.")
    # NEW: Separate directories for each algorithm
    parser.add_argument("--jmipod_dir", type=str,
                        default=str(Path.home() / "ALASKA2" / "Stego_small" / "JMiPOD"),
                        help="Path to JMiPOD stego images")
    parser.add_argument("--juniward_dir", type=str,
                        default=str(Path.home() / "ALASKA2" / "Stego_small" / "JUNIWARD"),
                        help="Path to JUNIWARD stego images")
    parser.add_argument("--uerd_dir", type=str,
                        default=str(Path.home() / "ALASKA2" / "Stego_small" / "UERD"),
                        help="Path to UERD stego images")
    # Legacy option
    parser.add_argument("--stego_dir", type=str, default=str(Path.home() / "ALASKA2" / "Stego_small"),
                        help="[DEPRECATED] Path to single stego directory")
    parser.add_argument("--use_balanced", action="store_true", default=True,
                        help="Use balanced multi-algorithm dataset (JMiPOD, JUNIWARD, UERD)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save the trained classifier.")
    # Training Args
    parser.add_argument("--batch_size", type=int, default=2, help="Physical batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning.")
    parser.add_argument("--accumulation_steps", type=int, default=32, help="Number of steps to accumulate gradients before an optimizer step. Effective batch size is batch_size * accumulation_steps.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio.")
    parser.add_argument("--num_workers", type=int, default=4, help="Num workers for data loading.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # --- Data Setup with Balanced Multi-Algorithm ---
    print("\n" + "="*70)
    print("BALANCED MULTI-ALGORITHM FINE-TUNING DATASET")
    print("="*70)

    if args.use_balanced:
        print("Building balanced dataset from 3 algorithms (JMiPOD, JUNIWARD, UERD)...")

        cover_dir_path = Path(args.cover_dir)
        jmipod_dir_path = Path(args.jmipod_dir)
        juniward_dir_path = Path(args.juniward_dir)
        uerd_dir_path = Path(args.uerd_dir)

        all_cover_paths = sorted(list(cover_dir_path.glob("*.jpg")))

        paths = []
        labels = []
        algo_labels = []  # Track which algorithm (for debugging)

        print(f"Found {len(all_cover_paths)} potential cover images")

        # Collect all valid triplets
        for cover_path in all_cover_paths:
            # Check if all 3 stego versions exist
            jmipod_path = jmipod_dir_path / cover_path.name
            juniward_path = juniward_dir_path / cover_path.name
            uerd_path = uerd_dir_path / cover_path.name

            if jmipod_path.is_file() and juniward_path.is_file() and uerd_path.is_file():
                # Add cover (label 0)
                paths.append(cover_path)
                labels.append(0)
                algo_labels.append(-1)  # -1 for cover

                # Add all 3 stego variants (label 1)
                for algo_idx, stego_path in enumerate([jmipod_path, juniward_path, uerd_path]):
                    paths.append(stego_path)
                    labels.append(1)
                    algo_labels.append(algo_idx)  # 0=JMiPOD, 1=JUNIWARD, 2=UERD

        # Report statistics
        total_covers = labels.count(0)
        total_stegos = labels.count(1)
        jmipod_count = algo_labels.count(0)
        juniward_count = algo_labels.count(1)
        uerd_count = algo_labels.count(2)

        print(f"\nDataset Statistics:")
        print(f"  - Total Cover: {total_covers}")
        print(f"  - Total Stego: {total_stegos}")
        print(f"    - JMiPOD: {jmipod_count}")
        print(f"    - JUNIWARD: {juniward_count}")
        print(f"    - UERD: {uerd_count}")
        print(f"  - Stego Ratio: {jmipod_count}:{juniward_count}:{uerd_count} (should be 1:1:1)")
        print(f"  - Cover:Stego = 1:{total_stegos//total_covers}")
        print("="*70 + "\n")
    else:
        print("Using SINGLE-ALGORITHM dataset (legacy mode)")
        cover_paths = list(Path(args.cover_dir).glob("*.jpg"))
        stego_paths = list(Path(args.stego_dir).glob("*.jpg"))
        if not cover_paths or not stego_paths:
            print("Error: No images found in the specified directories. Please check the paths.")
            return

        paths = cover_paths + stego_paths
        labels = [0] * len(cover_paths) + [1] * len(stego_paths)
        algo_labels = [-1] * len(paths)  # No algorithm tracking

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=args.val_ratio, random_state=42, stratify=labels
    )

    ds_train = ImageDataset(train_paths, train_labels)
    ds_val = ImageDataset(val_paths, val_labels)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Training with {len(ds_train)} samples, validating with {len(ds_val)} samples.")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")

    # --- Model Setup ---
    print("Initializing StegaNetRGB model...")
    model = StegaNetRGB().to(device)
    model_name = "steganet_rgb_finetuned_best.pth"

    if os.path.isfile(args.pretrained_path):
        print(f"\n{'='*70}")
        print(f"Loading pretrained weights from: {args.pretrained_path}")
        print(f"{'='*70}")
        try:
            # Load the state dict from the CAE pretraining.
            pretrained_dict = torch.load(args.pretrained_path, map_location=device)

            # Filter out incompatible keys (from MSCAE_RGB_CAE that don't exist in StegaNetRGB)
            model_dict = model.state_dict()
            incompatible_keys = []
            compatible_keys = []

            for k, v in pretrained_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    compatible_keys.append(k)
                else:
                    incompatible_keys.append(k)

            # Create filtered dict with only compatible weights
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in compatible_keys}

            # Load compatible weights
            missing_keys, _ = model.load_state_dict(filtered_dict, strict=False)

            print(f"\n✓ Successfully loaded {len(compatible_keys)} compatible layers:")
            # Group by module for cleaner output
            stage1_keys = [k for k in compatible_keys if 'stage1' in k]
            stage2_keys = [k for k in compatible_keys if 'stage2' in k]
            stage3_keys = [k for k in compatible_keys if 'stage3' in k]
            decoder_keys = [k for k in compatible_keys if 'decoder' in k]
            other_keys = [k for k in compatible_keys if not any(x in k for x in ['stage1', 'stage2', 'stage3', 'decoder'])]

            if stage1_keys:
                print(f"  - Stage1 Encoder: {len(stage1_keys)} layers")
            if stage2_keys:
                print(f"  - Stage2 Encoder: {len(stage2_keys)} layers")
            if stage3_keys:
                print(f"  - Stage3 Encoder: {len(stage3_keys)} layers")
            if decoder_keys:
                print(f"  - Decoder: {len(decoder_keys)} layers")
            if other_keys:
                print(f"  - Other: {len(other_keys)} layers")

            print(f"\n⚠ Skipped {len(incompatible_keys)} incompatible layers from pretrained CAE:")
            # Show first few incompatible keys as examples
            for k in incompatible_keys[:5]:
                print(f"  - {k}")
            if len(incompatible_keys) > 5:
                print(f"  ... and {len(incompatible_keys) - 5} more")

            print(f"\n⚠ Randomly initialized {len(missing_keys)} new layers (classifier head):")
            for k in list(missing_keys)[:5]:
                print(f"  - {k}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")

            print(f"\n{'='*70}")
            print(f"✓ Pretrained encoder loaded successfully!")
            print(f"  → Encoder: Initialized from pretrained CAE")
            print(f"  → Classifier Head: Randomly initialized (will be trained)")
            print(f"{'='*70}\n")

        except Exception as e:
            print(f"\n✗ Error loading pretrained weights: {e}")
            print("Training from scratch...\n")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠ Pretrained model not found at: {args.pretrained_path}")
        print("Training from scratch...\n")

    # --- Freeze MSCAE Encoder/Decoder, Train Classifier ONLY ---
    print(f"\n{'='*70}")
    print(f"🔒 FREEZING ENCODER/DECODER (Pretrained MSCAE)")
    print(f"{'='*70}")

    # Freeze all SRM filters
    for param in model.srm_branches.parameters():
        param.requires_grad = False
    print(f"  ✓ SRM Branches: FROZEN")

    # Freeze all encoder branches (Stage1 per branch)
    for param in model.encoder_branches.parameters():
        param.requires_grad = False
    print(f"  ✓ Encoder Branches (Stage1): FROZEN")

    # Freeze Stage2 and Stage3
    for param in model.stage2.parameters():
        param.requires_grad = False
    for param in model.stage3.parameters():
        param.requires_grad = False
    print(f"  ✓ Stage2 & Stage3: FROZEN")

    # Freeze decoder
    for param in model.decoder.parameters():
        param.requires_grad = False
    print(f"  ✓ Decoder: FROZEN")

    # Freeze channel selector
    for param in model.channel_selector.parameters():
        param.requires_grad = False
    print(f"  ✓ Channel Selector: FROZEN")

    # Freeze attention network (not used but still exists)
    for param in model.stego_discriminator.parameters():
        param.requires_grad = False
    model.attention_scale.requires_grad = False
    print(f"  ✓ Attention Network: FROZEN (not used)")

    print(f"\n🔥 TRAINING CLASSIFIER ONLY")
    print(f"{'='*70}")

    # Collect only trainable parameters (fc1, fc2)
    trainable_params = []
    trainable_param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            trainable_param_names.append(name)

    print(f"  ✓ Trainable Parameters: {len(trainable_param_names)}")
    for name in trainable_param_names:
        print(f"    - {name}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)
    frozen_param_count = total_params - trainable_param_count

    print(f"\n  📊 Parameter Statistics:")
    print(f"    - Total Parameters: {total_params:,}")
    print(f"    - Trainable (Classifier): {trainable_param_count:,} ({100*trainable_param_count/total_params:.2f}%)")
    print(f"    - Frozen (Encoder/Decoder): {frozen_param_count:,} ({100*frozen_param_count/total_params:.2f}%)")
    print(f"{'='*70}\n")

    # Optimizer with ONLY trainable parameters
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    print(f"✨ Training Configuration:")
    print(f"  - Learning Rate: {args.lr:.2e}")
    print(f"  - Weight Decay: 1e-4")
    print(f"  - Scheduler: CosineAnnealingLR (T_max={args.epochs})")
    print()

    scaler = GradScaler(enabled=device.type == "cuda")

    print(f"\n{'='*100}")
    print(f"{'EPOCH':<8} {'TRAIN LOSS':<12} {'TRAIN ACC':<12} {'TR RATIO':<10} {'VAL LOSS':<12} {'VAL ACC':<12} {'VAL AUC':<10} {'VAL RATIO':<10}")
    print(f"{'='*100}")

    best_val_auc = 0.0
    best_epoch = 0
    training_history = []

    for ep in range(1, args.epochs + 1):
        # Training with reconstruction error monitoring
        tr_loss, tr_acc, tr_recon_ratio = run_epoch(
            model, loader_train, optimizer, device, scaler, True,
            args.accumulation_steps
        )

        # Validation with reconstruction error monitoring
        val_loss, val_acc, val_auc, recon_stats = evaluate(model, loader_val, device)

        # Store history
        training_history.append({
            'epoch': ep,
            'train_loss': tr_loss,
            'train_acc': tr_acc,
            'train_recon_ratio': tr_recon_ratio,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'recon_cover_mean': recon_stats['cover_mean'],
            'recon_stego_mean': recon_stats['stego_mean'],
            'recon_ratio': recon_stats['ratio'],
            'lr': optimizer.param_groups[0]['lr']
        })

        # Print epoch summary
        print(f"{ep:<8} {tr_loss:<12.4f} {tr_acc:<12.4f} {tr_recon_ratio:<10.3f} {val_loss:<12.4f} {val_acc:<12.4f} {val_auc:<10.4f} {recon_stats['ratio']:<10.3f}")

        # Detailed reconstruction error stats
        if ep % 5 == 0 or ep == 1:  # Print detailed stats every 5 epochs
            attn_scale_val = model.attention_scale.item()
            print(f"  └─ Recon Error → Cover: {recon_stats['cover_mean']:.4f}±{recon_stats['cover_std']:.4f} | "
                  f"Stego: {recon_stats['stego_mean']:.4f}±{recon_stats['stego_std']:.4f} | "
                  f"Ratio: {recon_stats['ratio']:.3f}x | AttentionScale: {attn_scale_val:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = ep
            save_path = os.path.join(args.save_dir, model_name)
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'recon_stats': recon_stats
            }, save_path)
            print(f"  ✓ New best! Saved to {save_path}")

        scheduler.step()

    # Final summary
    print(f"{'='*100}")
    print(f"\n✓ Training Complete!")
    print(f"  → Best Validation AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    print(f"  → Model saved to: {os.path.join(args.save_dir, model_name)}")
    print(f"{'='*100}\n")

    # Save training history
    history_path = os.path.join(args.save_dir, "training_history.npy")
    np.save(history_path, training_history)
    print(f"Training history saved to: {history_path}")

if __name__ == "__main__":
    main()

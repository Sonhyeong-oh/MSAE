# -*- coding: utf-8 -*-
"""
MSCAE Contrastive & Reconstruction Pretraining Script (Stage 1)
- Pre-trains the Stage1 CAE using both contrastive and reconstruction losses.
- Contrastive loss pushes Cover and Stego embeddings apart.
- Reconstruction loss ensures features are still meaningful for image structure.
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
from torch.utils.data import Dataset, DataLoader, random_split
import random
from torch.cuda.amp import autocast, GradScaler

# Import both the model and the loss functions
from MSCAE_RGBmodel import MSCAE_RGB_CAE, SNRLoss

# --- NEW: Balanced Multi-Algorithm Dataset ---
class BalancedMultiAlgoDataset(Dataset):
    """
    Dataset that loads Cover images paired with 3 stego algorithms (JMiPOD, JUNIWARD, UERD)
    in a balanced 1:1:1 ratio. Supports stratified train/val splitting.
    """
    def __init__(self, cover_dir: str, jmipod_dir: str, juniward_dir: str, uerd_dir: str, use_cache: bool = True):
        cover_dir_path = Path(cover_dir)
        jmipod_dir_path = Path(jmipod_dir)
        juniward_dir_path = Path(juniward_dir)
        uerd_dir_path = Path(uerd_dir)

        print("\n" + "="*70)
        print("BALANCED MULTI-ALGORITHM DATASET")
        print("="*70)
        print("Scanning cover directory...")
        all_cover_paths = sorted(list(cover_dir_path.glob("*.jpg")))

        self.cover_paths = []
        self.stego_paths = []
        self.algo_labels = []  # 0=JMiPOD, 1=JUNIWARD, 2=UERD

        algo_dirs = {
            'JMiPOD': jmipod_dir_path,
            'JUNIWARD': juniward_dir_path,
            'UERD': uerd_dir_path
        }

        print(f"Found {len(all_cover_paths)} potential cover images.")
        print("Matching with stego algorithms (JMiPOD, JUNIWARD, UERD)...\n")

        for cover_path in tqdm(all_cover_paths, desc="Building balanced dataset"):
            # Try to find matching stego in all 3 algorithms
            algo_found = {}

            for algo_name, algo_dir in algo_dirs.items():
                stego_path = algo_dir / cover_path.name
                if stego_path.is_file():
                    algo_found[algo_name] = stego_path

            # Only include if ALL 3 algorithms exist for this cover
            if len(algo_found) == 3:
                # Add one entry per algorithm (3 entries per cover)
                for algo_idx, (algo_name, stego_path) in enumerate(sorted(algo_found.items())):
                    self.cover_paths.append(cover_path)
                    self.stego_paths.append(stego_path)
                    self.algo_labels.append(algo_idx)  # 0=JMiPOD, 1=JUNIWARD, 2=UERD

        if not self.cover_paths:
            raise FileNotFoundError(
                f"Could not find matching image triplets (Cover + 3 stego algorithms).\n"
                f"Please ensure each cover has JMiPOD, JUNIWARD, and UERD versions."
            )

        # Report statistics
        total_triplets = len(self.cover_paths) // 3
        algo_counts = {0: 0, 1: 0, 2: 0}
        for label in self.algo_labels:
            algo_counts[label] += 1

        print(f"\n{'='*70}")
        print(f"Dataset Statistics:")
        print(f"  - Total Cover Images: {total_triplets}")
        print(f"  - Total Pairs (Cover-Stego): {len(self.cover_paths)}")
        print(f"  - JMiPOD pairs: {algo_counts[0]}")
        print(f"  - JUNIWARD pairs: {algo_counts[1]}")
        print(f"  - UERD pairs: {algo_counts[2]}")
        print(f"  - Ratio: {algo_counts[0]}:{algo_counts[1]}:{algo_counts[2]} (should be 1:1:1)")
        print(f"{'='*70}\n")

    def __len__(self):
        return len(self.cover_paths)

    def __getitem__(self, idx):
        try:
            # Load cover image
            cover_img = Image.open(self.cover_paths[idx]).convert("RGB")
            cover_img = cover_img.resize((256, 256), Image.Resampling.LANCZOS)
            cover_np = np.array(cover_img, dtype=np.float32)
            cover = torch.from_numpy(cover_np.transpose((2, 0, 1)))

            # Load stego image
            stego_img = Image.open(self.stego_paths[idx]).convert("RGB")
            stego_img = stego_img.resize((256, 256), Image.Resampling.LANCZOS)
            stego_np = np.array(stego_img, dtype=np.float32)
            stego = torch.from_numpy(stego_np.transpose((2, 0, 1)))

            # Algorithm label
            algo_label = torch.tensor(self.algo_labels[idx], dtype=torch.long)

            return cover, stego, algo_label
        except Exception as e:
            print(f"Error loading image pair at index {idx}: {e}")
            return torch.zeros(3, 256, 256), torch.zeros(3, 256, 256), torch.tensor(0, dtype=torch.long)


class ContrastiveDataset(Dataset):
    """
    Dataset class that loads pairs of (cover, stego) images for contrastive pre-training.
    This version explicitly matches filenames between cover and stego directories
    to ensure correct pairing.
    """
    def __init__(self, cover_dir: str, stego_dir: str, use_cache: bool = True):
        cover_dir_path = Path(cover_dir)
        stego_dir_path = Path(stego_dir)

        print("Scanning cover directory for JPG files...")
        all_cover_paths = list(cover_dir_path.glob("*.jpg"))

        self.cover_paths = []
        self.stego_paths = []

        print(f"Found {len(all_cover_paths)} potential cover images. Now matching with stego images...")
        for cover_path in tqdm(all_cover_paths, desc="Matching pairs"):
            # For new dataset: cover and stego filenames are identical (e.g., 00001.jpg)
            stego_path = stego_dir_path / cover_path.name
            if stego_path.is_file():
                self.cover_paths.append(cover_path)
                self.stego_paths.append(stego_path)

        if not self.cover_paths:
            raise FileNotFoundError(
                f"Could not find any matching image pairs between "
                f"'{cover_dir}' and '{stego_dir}'.\n"
                f"Please ensure that for a cover image like 'cover/00001.jpg', a corresponding 'stego/00001.jpg' exists."
            )

        print(f"Successfully found {len(self.cover_paths)} matching cover/stego image pairs.")

    def _get_paths(self, directory: Path, use_cache: bool, cache_name: str) -> list:
        cache_file = directory / cache_name

        if use_cache and cache_file.is_file():
            print(f"Loading file list from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                # Return full absolute paths
                paths = [directory / Path(line.strip()) for line in f.readlines()]
            return paths

        print(f"Scanning directory: {directory}. This may take a moment...")
        paths = list(directory.glob("*.jpg"))
        
        if use_cache:
            print(f"Saving file list to cache for future runs: {cache_file}")
            with open(cache_file, 'w') as f:
                # Save only the filenames, not the full path
                for path in paths:
                    f.write(f"{path.name}\n")
        return paths

    def __len__(self):
        return len(self.cover_paths)

    def __getitem__(self, idx):
        try:
            # Load cover image as RGB
            cover_img = Image.open(self.cover_paths[idx]).convert("RGB")
            cover_img = cover_img.resize((256, 256), Image.Resampling.LANCZOS)
            # NO normalization - keep [0, 255] range for stronger signal
            cover_np = np.array(cover_img, dtype=np.float32)
            cover = torch.from_numpy(cover_np.transpose((2, 0, 1))) # HWC to CHW

            # Load corresponding stego image as RGB
            stego_img = Image.open(self.stego_paths[idx]).convert("RGB")
            stego_img = stego_img.resize((256, 256), Image.Resampling.LANCZOS)
            # NO normalization - keep [0, 255] range for stronger signal
            stego_np = np.array(stego_img, dtype=np.float32)
            stego = torch.from_numpy(stego_np.transpose((2, 0, 1))) # HWC to CHW
            
            return cover, stego
        except Exception as e:
            print(f"Error loading image pair at index {idx}: {self.cover_paths[idx]}")
            print(e)
            # Return 3-channel zero tensors on error
            return torch.zeros(3, 256, 256), torch.zeros(3, 256, 256)

# --- UPDATED: Training loop with SNR Loss ---
def run_epoch_snr(model, loader, optimizer, device, scaler, snr_loss_fn, recon_weight, snr_weight, use_amp, accumulation_steps=1):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss_sum = 0.0
    snr_loss_sum = 0.0
    recon_loss_sum = 0.0
    
    pbar = tqdm(loader, desc="Training" if is_training else "Validation", leave=False)

    if is_training:
        optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        # Support both old format (cover, stego) and new format (cover, stego, algo_label)
        if len(batch) == 2:
            cover_imgs, stego_imgs = batch
            algo_labels = None
        else:
            cover_imgs, stego_imgs, algo_labels = batch

        cover_imgs = cover_imgs.to(device, non_blocking=True)
        stego_imgs = stego_imgs.to(device, non_blocking=True)
        if algo_labels is not None:
            algo_labels = algo_labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            # 1. Get model outputs for both cover and stego images
            # is_cover=True updates the running stats for cover images
            x_hat_cover, z_cover_map, x_filtered_cover, cover_amplified_diff = model(cover_imgs, is_cover=True)
            # is_cover=False uses the stats for amplification but does not update them
            x_hat_stego, z_stego_map, x_filtered_stego, stego_amplified_diff = model(stego_imgs, is_cover=False)

            # 2. Calculate Amplified Reconstruction Loss for COVER images only.
            # This trains the model to be an expert at reconstructing covers and sensitive to anomalies.
            if cover_imgs.numel() > 0:
                recon_loss = torch.mean(cover_amplified_diff)
            else:
                recon_loss = torch.tensor(0.0, device=device)

            # 3. Calculate SNR Loss (Stego noise should be higher than Cover noise)
            if cover_amplified_diff.numel() > 0 and stego_amplified_diff.numel() > 0:
                snr_loss = snr_loss_fn(cover_amplified_diff, stego_amplified_diff)
            else:
                snr_loss = torch.tensor(0.0, device=device)

            # 4. Total Loss (weighted combination)
            total_loss = snr_weight * snr_loss + recon_weight * recon_loss
            if is_training:
                total_loss = total_loss / accumulation_steps

        if is_training and torch.isfinite(total_loss):
            scaler.scale(total_loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        batch_size = cover_imgs.size(0)
        scale_factor = (accumulation_steps if is_training else 1)
        total_loss_sum += total_loss.item() * batch_size * scale_factor
        snr_loss_sum += snr_loss.item() * batch_size
        recon_loss_sum += recon_loss.item() * batch_size

        pbar.set_postfix({
            "total_loss": f"{total_loss.item() * scale_factor:.4f}",
            "snr_loss": f"{snr_loss.item():.4f}",
            "recon_loss": f"{recon_loss.item():.4f}"
        })

    n_samples = len(loader.dataset)
    return (total_loss_sum / n_samples, snr_loss_sum / n_samples, recon_loss_sum / n_samples)

def main():
    parser = argparse.ArgumentParser(description="MSCAE RGB Contrastive Pretraining with Balanced Multi-Algorithm Dataset")
    parser.add_argument("--cover_dir", type=str,
                        default=str(Path.home() / "ALASKA2" / "Cover_small"),
                        help="Path to cover images directory")
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
    # Legacy option (ignored if using balanced dataset)
    parser.add_argument("--stego_dir", type=str,
                        default=str(Path.home() / "ALASKA2" / "Stego_small"),
                        help="[DEPRECATED] Path to single stego directory (use --jmipod_dir, --juniward_dir, --uerd_dir instead)")
    parser.add_argument("--use_balanced", action="store_true", default=True,
                        help="Use balanced multi-algorithm dataset (JMiPOD, JUNIWARD, UERD)")
    parser.add_argument("--recon_weight", type=float, default=1.0, help="Weight for the reconstruction loss component.")
    parser.add_argument("--snr_weight", type=float, default=100.0, help="Weight for the SNR loss component (higher = prioritize discrimination).")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--save_path", type=str, default="./checkpoints_cae/cae_rgb_balanced_stage1.pth", help="Path to save the pretrained model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--accumulation_steps", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--no_cache", action="store_true", help="Force re-scanning of data directories instead of using cache.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Channel-wise amplification factor for reconstruction error (recommended: 0.1-0.2).")
    parser.add_argument("--spatial_alpha", type=float, default=0.15, help="Spatial amplification factor for patch-level anomaly detection (recommended: 0.05-0.1).")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size for spatial statistics tracking (default: 32x32).")
    args = parser.parse_args()

    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Create balanced multi-algorithm dataset
    if args.use_balanced:
        print("Using BALANCED MULTI-ALGORITHM dataset")
        dataset = BalancedMultiAlgoDataset(
            cover_dir=args.cover_dir,
            jmipod_dir=args.jmipod_dir,
            juniward_dir=args.juniward_dir,
            uerd_dir=args.uerd_dir,
            use_cache=not args.no_cache
        )
    else:
        print("Using SINGLE-ALGORITHM dataset (legacy mode)")
        dataset = ContrastiveDataset(args.cover_dir, args.stego_dir, use_cache=not args.no_cache)

    # Stratified split to maintain algorithm balance
    val_len = int(len(dataset) * 0.1)
    train_len = len(dataset) - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- UPDATED: Instantiate RGB model and loss functions ---
    model = MSCAE_RGB_CAE(alpha=args.alpha, spatial_alpha=args.spatial_alpha, patch_size=args.patch_size).to(device)
    # RATIO-BASED SNR Loss (not embedding-aware) for train/test consistency
    snr_loss_fn = SNRLoss(target_ratio=1.5, loss_type='ratio', use_embedding_diff=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n{'='*60}")
    print(f"Model Configuration:")
    print(f"  - Channel Alpha: {args.alpha} (EXPONENTIAL amplification: exp(alpha * |z_score|))")
    print(f"  - Spatial Alpha: {args.spatial_alpha} (EXPONENTIAL amplification for spatial anomalies)")
    print(f"  - Patch Size: {args.patch_size}x{args.patch_size} (spatial statistics grid)")
    print(f"  - Image Size: 256x256 -> {256//args.patch_size}x{256//args.patch_size} patch grid")
    print(f"\nLoss Configuration:")
    print(f"  - SNR Mode: Ratio-based (stego_noise / cover_noise >= {snr_loss_fn.target_ratio})")
    print(f"  - Amplification: Exponential (small deviations → mild boost, large deviations → aggressive boost)")
    print(f"  - SNR Weight: {args.snr_weight}x (discrimination priority)")
    print(f"  - Recon Weight: {args.recon_weight}x (reconstruction priority)")
    print(f"  - Effective ratio: SNR/{args.snr_weight/args.recon_weight:.1f}x more important than Recon")
    print(f"{'='*60}\n")

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_total_loss, train_snr_loss, train_recon_loss = run_epoch_snr(
            model, train_loader, optimizer, device, scaler, snr_loss_fn, args.recon_weight, args.snr_weight, use_amp, args.accumulation_steps
        )
        with torch.no_grad():
            val_total_loss, val_snr_loss, val_recon_loss = run_epoch_snr(
                model, val_loader, None, device, None, snr_loss_fn, args.recon_weight, args.snr_weight, False, 1
            )

        print(f"Epoch {epoch:2d}/{args.epochs} | Train Total: {train_total_loss:.4f} (SNR: {train_snr_loss:.4f}, Recon: {train_recon_loss:.4f}) "
              f"| Val Total: {val_total_loss:.4f} (SNR: {val_snr_loss:.4f}, Recon: {val_recon_loss:.4f})")

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Saved best model to {args.save_path} (Val Loss: {best_val_loss:.4f})")

    print("\nContrastive Pretraining Completed!")

if __name__ == "__main__":
    main()
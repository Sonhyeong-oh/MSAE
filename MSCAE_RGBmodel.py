# -*- coding: utf-8 -*-
"""
MSCAE Model Architectures for Steganalysis (RGB VERSION)
Multi-branch Stacked Convolutional Auto-Encoders for three-channel (RGB) images.
- SRM filters are applied to each RGB channel independently.
- Designed for color image steganalysis.
- Stage3 uses a Dual Attention mechanism.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =====================================================================
# Helper Modules
# =====================================================================

class ChannelPooling(nn.Module):
    """
    Channel-wise pooling for Spatial Attention (CBAM-style).
    Compresses feature maps along channel dimension using Max and Avg pooling.
    """
    def forward(self, x):
        # Max pooling across channel dimension
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        # Avg pooling across channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)    # (B, 1, H, W)
        # Concatenate along channel dimension
        return torch.cat([max_pool, avg_pool], dim=1)    # (B, 2, H, W)


# =====================================================================
# Filter Initialization Functions
# =====================================================================
def srm_filters(scale_for_normalized_input=False):
    """
    SRM (Spatial Rich Model) filters for steganalysis.

    Returns 30 diverse 5x5 kernels that are effective for steganalysis.
    These are the base filters that will be applied to each RGB channel.
    """
    filters = []

    # 1st Order
    filters.append([[0,0,0,0,0],[0,0,0,0,0],[-1,2,-2,2,-1],[0,0,0,0,0],[0,0,0,0,0]])
    filters.append([[0,0,-1,0,0],[0,0,2,0,0],[0,0,-2,0,0],[0,0,2,0,0],[0,0,-1,0,0]])
    filters.append([[-1,2,0,0,0],[2,-2,0,0,0],[0,0,0,0,0],[0,0,0,-2,2],[0,0,0,2,-1]])
    filters.append([[0,0,0,2,-1],[0,0,0,-2,2],[0,0,0,0,0],[2,-2,0,0,0],[-1,2,0,0,0]])
    filters.append([[0,0,-1,0,0],[0,0,3,0,0],[0,0,-3,0,0],[0,0,3,0,0],[0,0,-2,0,0]])
    filters.append([[0,0,0,0,0],[0,0,0,0,0],[-1,3,-3,3,-2],[0,0,0,0,0],[0,0,0,0,0]])
    filters.append([[-1,3,0,0,0],[3,-3,0,0,0],[0,0,0,0,0],[0,0,0,-3,3],[0,0,0,3,-2]])
    filters.append([[0,0,0,3,-1],[0,0,0,-3,3],[0,0,0,0,0],[3,-3,0,0,0],[-2,3,0,0,0]])

    # 2nd Order
    filters.append([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]])
    filters.append([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]])
    filters.append([[0,0,-1,0,0],[0,0,2,0,0],[-1,2,-4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]])
    filters.append([[-1,0,0,0,-1],[0,2,0,2,0],[0,0,-4,0,0],[0,2,0,2,0],[-1,0,0,0,-1]])
    filters.append([[0,-1,-1,-1,0],[-1,2,2,2,-1],[-1,2,-8,2,-1],[-1,2,2,2,-1],[0,-1,-1,-1,0]])
    filters.append([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]])
    filters.append([[1,-2,0,-2,1],[-2,4,0,4,-2],[0,0,0,0,0],[-2,4,0,4,-2],[1,-2,0,-2,1]])
    filters.append([[0,0,-2,0,0],[0,4,0,4,0],[-2,0,-8,0,-2],[0,4,0,4,0],[0,0,-2,0,0]])

    # 3rd Order
    filters.append([[0,0,1,0,0],[0,-2,0,-2,0],[1,0,2,0,1],[0,-2,0,-2,0],[0,0,1,0,0]])
    filters.append([[0,1,0,1,0],[1,-4,0,-4,1],[0,0,8,0,0],[1,-4,0,-4,1],[0,1,0,1,0]])
    filters.append([[-1,2,-1,2,-1],[2,-4,2,-4,2],[-1,2,-2,2,-1],[2,-4,2,-4,2],[-1,2,-1,2,-1]])
    filters.append([[1,-2,1,-2,1],[-2,4,-2,4,-2],[1,-2,2,-2,1],[-2,4,-2,4,-2],[1,-2,1,-2,1]])
    filters.append([[0,0,2,0,0],[0,-4,0,-4,0],[2,0,-8,0,2],[0,-4,0,-4,0],[0,0,2,0,0]])
    filters.append([[-1,0,2,0,-1],[0,2,0,2,0],[2,0,-8,0,2],[0,2,0,2,0],[-1,0,2,0,-1]])
    filters.append([[1,0,-2,0,1],[0,-2,0,-2,0],[-2,0,8,0,-2],[0,-2,0,-2,0],[1,0,-2,0,1]])
    filters.append([[0,-1,0,-1,0],[-1,4,0,4,-1],[0,0,-8,0,0],[-1,4,0,4,-1],[0,-1,0,-1,0]])

    # SQUARE Filters
    filters.append([[2,-1,-1,-1,2],[-1,2,-1,2,-1],[-1,-1,0,-1,-1],[-1,2,-1,2,-1],[2,-1,-1,-1,2]])
    filters.append([[-1,2,-1,2,-1],[2,-4,2,-4,2],[-1,2,-2,2,-1],[2,-4,2,-4,2],[-1,2,-1,2,-1]])
    filters.append([[1,-2,1,-2,1],[-2,4,-2,4,-2],[1,-2,2,-2,1],[-2,4,-2,4,-2],[1,-2,1,-2,1]])

    # EDGE Filters
    filters.append([[0,0,-1,0,0],[0,-1,4,-1,0],[-1,4,-12,4,-1],[0,-1,4,-1,0],[0,0,-1,0,0]])
    filters.append([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])
    filters.append([[1,1,1,1,1],[1,-2,-2,-2,1],[1,-2,-8,-2,1],[1,-2,-2,-2,1],[1,1,1,1,1]])

    filters_np = np.array(filters, dtype=np.float32)
    for i in range(filters_np.shape[0]):
        fsum = np.abs(filters_np[i]).sum()
        if fsum > 0:
            filters_np[i] = filters_np[i] / fsum
    if scale_for_normalized_input:
        filters_np = filters_np * 255.0
    return torch.from_numpy(filters_np)

# =====================================================================
# Core Building Blocks (RGB Adapted)
# =====================================================================

class ResidualStage(nn.Module):
    """
    Residual Stage with skip connections for improved gradient flow.
    This version is generic and works with any number of input channels.

    Uses GroupNorm instead of BatchNorm for batch-size independence.
    """
    def __init__(self, in_ch: int, maps_per_input: int, k: int = 3, pool: int = 4):
        super().__init__()
        padding = k // 2
        out_ch = in_ch * maps_per_input
        groups = 1  # Use standard convolutions for better gradient flow

        # Determine optimal number of groups for GroupNorm
        # Target: 8-16 channels per group
        num_groups = max(out_ch // 16, 1)  # At least 1 group
        # Ensure num_groups divides out_ch evenly
        while out_ch % num_groups != 0:
            num_groups -= 1
        num_groups = max(num_groups, 1)

        # Main path
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=padding, groups=groups, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=padding, groups=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)

        # Skip connection to handle channel dimension changes
        self.skip_connection = None
        if in_ch != out_ch:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
            )

        self.pool = nn.MaxPool2d(kernel_size=pool, stride=pool)
        self.out_ch = out_ch

    def forward(self, x):
        identity = x
        # Main path
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.gn2(out)
        # Skip connection
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
        out = out + identity
        out = F.relu(out, inplace=True)
        # Pooling
        out = self.pool(out)
        return out


class DualAttentionStage(nn.Module):
    """
    Dual Attention Stage for adaptive stego noise detection.
    This version is generic and works with any number of input channels.
    """
    def __init__(self, in_ch: int, maps_per_input: int = 10, k: int = 5, pool: int = 4,
                 reduction_ratio: int = 16, spatial_kernel: int = 7):
        super().__init__()
        padding = k // 2
        out_ch = in_ch * maps_per_input
        groups = in_ch if in_ch > 1 else 1 # Depthwise convolution

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=padding, groups=groups, bias=True)
        
        # Channel Attention
        reduced_ch = max(out_ch // reduction_ratio, 1)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, reduced_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, out_ch, 1),
            nn.Sigmoid()
        )

        # Spatial Attention
        spatial_padding = spatial_kernel // 2
        self.spatial_attn = nn.Sequential(
            ChannelPooling(),
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_padding),
            nn.Sigmoid()
        )

        self.pool = nn.MaxPool2d(kernel_size=pool, stride=pool)
        self.out_ch = out_ch

    def forward(self, x):
        feat = self.conv(x)
        feat = F.relu(feat, inplace=True)
        # Apply channel and spatial attention
        feat = feat * self.channel_attn(feat)
        feat = feat * self.spatial_attn(feat)
        out = self.pool(feat)
        return out


# =====================================================================
# Main Classification Model (RGB Version)
# =====================================================================

class StegaNetRGB(nn.Module):
    """
    Multi-branch StegaNet adapted for RGB images with Stego-Discriminative Attention.

    Architecture:
    - Applies SRM filters to each RGB channel independently
    - Uses a hybrid classifier with encoded features and reconstruction error

    Key Components:
    1. SRM Filtering (90 channels)
    2. Encoder (Stage1 → Stage2 → Stage3 with DualAttention)
    3. Decoder (for reconstruction error)
    4. channel_selector: Chooses between max/avg pooling
    5. stego_discriminator: Learns which latent dimensions distinguish stego
    6. Hybrid Classifier: Concatenates amplified latent + reconstruction error features
    """
    def __init__(self):
        super().__init__()

        # ✨ UPDATED: Multi-Branch Architecture (matches MSCAE_RGB_CAE for pretrained weights!)
        # Get all 30 SRM filters
        srm_kernel = srm_filters(scale_for_normalized_input=False)  # (30, 5, 5)

        # Define branch configurations for SRM filters (same as MSCAE_RGB_CAE)
        self.branch_configs = [
            ('1st_order', 8, (0, 8)),      # 8 filters × 3 RGB = 24 channels
            ('2nd_order', 8, (8, 16)),     # 8 filters × 3 RGB = 24 channels
            ('3rd_order', 8, (16, 24)),    # 8 filters × 3 RGB = 24 channels
            ('sq_edge', 6, (24, 30))       # 6 filters × 3 RGB = 18 channels
        ]                                   # Total: 90 channels

        # Create SRM branches (applied to RGB input)
        self.srm_branches = nn.ModuleList()
        for _, num_filters, (start_idx, end_idx) in self.branch_configs:
            # Each branch: 3 RGB channels → (num_filters × 3) output channels
            srm_conv = nn.Conv2d(3, num_filters * 3, kernel_size=5, padding=2, groups=3, bias=False)
            with torch.no_grad():
                # Repeat the filter subset for each RGB channel
                srm_subset = srm_kernel[start_idx:end_idx]  # (num_filters, 5, 5)
                srm_subset_rgb = srm_subset.unsqueeze(1).repeat(3, 1, 1, 1)  # (3, num_filters, 5, 5)
                srm_conv.weight.copy_(srm_subset_rgb)
            self.srm_branches.append(srm_conv)

        # Create encoder branches (one ResidualStage per SRM branch) - Stage1 equivalent
        self.encoder_branches = nn.ModuleList()
        total_encoder_out_ch = 0
        for _, num_filters, _ in self.branch_configs:
            in_ch = num_filters * 3  # e.g., 8 filters × 3 RGB = 24 channels
            encoder = ResidualStage(in_ch=in_ch, maps_per_input=4, k=3, pool=4)
            self.encoder_branches.append(encoder)
            total_encoder_out_ch += encoder.out_ch

        # Encoder Stages (360 -> 1440 -> 2880)
        self.stage2 = ResidualStage(in_ch=total_encoder_out_ch, maps_per_input=4, k=3, pool=4)
        self.stage3 = DualAttentionStage(in_ch=self.stage2.out_ch, maps_per_input=2, k=3, pool=4)
        stage3_channels = 2880

        # Decoder for Reconstruction Error Feature
        # Calculate optimal GroupNorm groups
        num_groups_1440 = 90  # 1440 / 90 = 16 channels/group
        num_groups_360 = 45   # 360 / 45 = 8 channels/group

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(stage3_channels, 1440, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_1440, num_channels=1440), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(1440, 360, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_360, num_channels=360), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(360, 90, kernel_size=3, padding=1, bias=True),
        )

        # Classifier Head
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = stage3_channels * 2 // 32
        self.channel_selector = nn.Sequential(
            nn.Linear(stage3_channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, stage3_channels),
            nn.Sigmoid()
        )

        # Input: concatenated [encoded_pooled (2880) + error_features (90)]
        combined_input_dim = stage3_channels + 90  # 2880 + 90 = 2970
        self.stego_discriminator = nn.Sequential(
            nn.Linear(combined_input_dim, stage3_channels // 2),  # 2970 → 1440
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # Prevent overfitting
            nn.Linear(stage3_channels // 2, stage3_channels),  # 1440 → 2880
            nn.Sigmoid()  # Output range [0, 1] → will be scaled to [0.5, 1.5]
        )

        # Learnable amplification scale (initialized to 1.0 for [0.5, 1.5] range)
        self.attention_scale = nn.Parameter(torch.ones(1))

        # Hybrid Classifier with Variance-Amplified Error Features
        # Input: encoded_pooled (2880) + variance-amplified error (90) = 2970
        self.fc1 = nn.Linear(stage3_channels + 90, 2048)  # 2970 → 2048
        self.fc2 = nn.Linear(2048, 2)

    def forward(self, x):
        # 1. Multi-Branch SRM Filtering
        filtered_branches = [srm_branch(x) for srm_branch in self.srm_branches]

        # 2. Multi-Branch Encoding (Stage1 equivalent)
        encoded_branches = [encoder(filtered) for encoder, filtered in zip(self.encoder_branches, filtered_branches)]

        # Concatenate all filtered outputs (for reconstruction comparison)
        x_filtered = torch.cat(filtered_branches, dim=1)  # (B, 90, H, W)

        # Concatenate all encoded outputs
        encoded = torch.cat(encoded_branches, dim=1)  # (B, 360, H/4, W/4)

        # 3. Stage2 and Stage3
        encoded = self.stage2(encoded)
        encoded = self.stage3(encoded)

        # 3. Decoder
        reconstructed = self.decoder(encoded)

        # 4. Reconstruction Error (for attention only, not classifier)
        diff_map = torch.abs(x_filtered - reconstructed)

        # 5. Adaptive Pooling of Encoded Features
        max_feat = self.max_pool(encoded).flatten(1)  # (B, 2880)
        avg_feat = self.avg_pool(encoded).flatten(1)  # (B, 2880)
        channel_weights = self.channel_selector(torch.cat([max_feat, avg_feat], dim=1))  # (B, 2880)
        encoded_pooled = channel_weights * max_feat + (1 - channel_weights) * avg_feat  # (B, 2880)

        # 6. SPATIAL VARIANCE AMPLIFICATION (Option 3)
        # Amplify high-variance regions in diff_map (edges/textures where stego hides)
        # NO learnable parameters - pure statistical computation for generalization

        B, C, H, W = diff_map.shape  # (B, 90, 64, 64)

        # Compute local variance using sliding window (8×8 patches)
        kernel_size = 8
        num_patches_h = H // kernel_size  # 64 // 8 = 8
        num_patches_w = W // kernel_size  # 64 // 8 = 8

        unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=kernel_size)
        patches = unfold(diff_map)  # (B, 90*8*8, 8*8) = (B, 5760, 64)

        # Reshape to (B, 90, 8*8, 64 patches) for per-channel variance
        patches = patches.view(B, C, kernel_size * kernel_size, num_patches_h * num_patches_w)  # (B, 90, 64, 64)

        # Variance per patch (across the 64 pixels in each 8×8 patch)
        # Average across channels for spatial variance map
        patch_variance = torch.var(patches, dim=2)  # (B, 90, 64)
        patch_variance = patch_variance.mean(dim=1, keepdim=True)  # (B, 1, 64) - Average over channels

        # Reshape to spatial grid (8×8 patches)
        variance_map = patch_variance.view(B, 1, num_patches_h, num_patches_w)  # (B, 1, 8, 8)

        # Upsample to match diff_map spatial size (64×64)
        variance_map_upsampled = F.interpolate(
            variance_map, size=(H, W), mode='bilinear', align_corners=False
        )  # (B, 1, 64, 64)

        # Normalize variance map to [0.5, 1.5] range
        # High variance (edges/textures) → 1.5x amplification
        # Low variance (smooth regions) → 0.5x suppression
        variance_normalized = variance_map_upsampled / (variance_map_upsampled.mean(dim=[2,3], keepdim=True) + 1e-6)
        amplification_map = 0.5 + torch.clamp(variance_normalized, 0, 2.0) * 0.5  # (B, 1, 64, 64) in [0.5, 1.5]

        # Apply spatial amplification to diff_map
        diff_map_amplified = diff_map * amplification_map  # (B, 90, 64, 64)

        # 7. Simple Error Features (NO hand-crafted 439 features!)
        # Extract compact error statistics from variance-amplified diff_map
        error_pooled = diff_map_amplified.mean(dim=[2, 3])  # (B, 90) - Mean per SRM channel

        # 8. Hybrid Classifier
        # Concatenate learned features + variance-amplified error features
        combined_features = torch.cat([encoded_pooled, error_pooled], dim=1)  # (B, 2970)

        # Classification head
        logits = F.relu(self.fc1(combined_features))  # (B, 2970) → (B, 2048)
        logits = self.fc2(logits)  # (B, 2048) → (B, 2)

        return logits, reconstructed

    def _compute_error_features(self, diff_map):
        B = diff_map.shape[0]
        
        # Global, Channel-wise, Patch-wise stats
        recon_error_mean = diff_map.mean(dim=[1, 2, 3])
        recon_error_max = diff_map.amax(dim=[1, 2, 3])
        recon_error_std = diff_map.std(dim=[1, 2, 3])
        channel_mean = diff_map.mean(dim=[2, 3])
        channel_max = diff_map.amax(dim=[2, 3])
        channel_std = diff_map.std(dim=[2, 3])
        channel_median = diff_map.median(dim=2)[0].median(dim=2)[0]
        
        H, W = diff_map.shape[2], diff_map.shape[3]
        patch_size = H // 8
        patches = diff_map.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patch_errors = patches.mean(dim=[1, 4, 5]).view(B, -1)

        # Percentile, Frequency, Concentration stats
        diff_flat = diff_map.view(B, -1)
        diff_flat_float = diff_flat.float()  # Ensure float32 for unsupported ops
        percentiles_q = torch.tensor([10, 25, 50, 75, 90, 95, 99], device=diff_map.device, dtype=torch.float32)
        percentile_values = torch.quantile(diff_flat_float, percentiles_q / 100.0, dim=1).T
        
        grad_x = torch.abs(diff_map[..., 1:] - diff_map[..., :-1])
        grad_y = torch.abs(diff_map[..., 1:, :] - diff_map[..., :-1, :])
        grad_mag_mean = (grad_x.mean(dim=[1, 2, 3]) + grad_y.mean(dim=[1, 2, 3])) / 2
        grad_mag_max = torch.maximum(grad_x.amax(dim=[1, 2, 3]), grad_y.amax(dim=[1, 2, 3]))
        grad_mag_std = (grad_x.std(dim=[1, 2, 3]) + grad_y.std(dim=[1, 2, 3])) / 2
        
        high_error_ratio = (diff_flat_float > percentile_values[:, 4:5]).float().mean(dim=1)
        
        histograms = [torch.histc(diff_flat_float[b], bins=10) for b in range(B)]
        histograms = torch.stack(histograms)
        probs = histograms / (histograms.sum(dim=1, keepdim=True) + 1e-10)
        error_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

        # Combine all 439 features
        return torch.cat([
            recon_error_mean.unsqueeze(1), recon_error_max.unsqueeze(1), recon_error_std.unsqueeze(1),
            channel_mean, channel_max, channel_std, channel_median,
            patch_errors, percentile_values,
            grad_mag_mean.unsqueeze(1), grad_mag_max.unsqueeze(1), grad_mag_std.unsqueeze(1),
            high_error_ratio.unsqueeze(1), error_entropy.unsqueeze(1),
        ], dim=1)

# =====================================================================
# Autoencoder & Contrastive Loss for Pretraining
# =====================================================================

class MSCAE_RGB_CAE(nn.Module):
    """
    Multi-branch Convolutional Autoencoder for pretraining the RGB model stages.
    Reconstructs the 90-channel SRM-filtered image from the latent vector.

    METHOD 2: LEARNABLE AMPLIFICATION
    - NO statistics tracking (solves Method 1's generalization problem!)
    - Amplification learned directly from data via SNR Loss gradient
    - 3-layer CNN predicts pixel-wise amplification factors
    - Better Train/Val/Test consistency (no dataset-specific statistics)

    MULTI-BRANCH ARCHITECTURE:
    - Splits 30 SRM filters into 4 parallel branches (1st_order, 2nd_order, 3rd_order, sq_edge)
    - Each RGB channel gets all 30 filters → 90 total channels split across branches
    - Gradient flows through amplification for end-to-end learning
    """
    def __init__(self, alpha=0.1, spatial_alpha=0.05, patch_size=32):
        super().__init__()

        # Get all 30 SRM filters
        srm_kernel = srm_filters(scale_for_normalized_input=False)  # (30, 5, 5)

        # Define branch configurations for SRM filters
        # Each branch gets a subset of the 30 filters
        self.branch_configs = [
            ('1st_order', 8, (0, 8)),      # 8 filters × 3 RGB = 24 channels
            ('2nd_order', 8, (8, 16)),     # 8 filters × 3 RGB = 24 channels
            ('3rd_order', 8, (16, 24)),    # 8 filters × 3 RGB = 24 channels
            ('sq_edge', 6, (24, 30))       # 6 filters × 3 RGB = 18 channels
        ]                                   # Total: 90 channels

        # Create SRM branches (applied to RGB input)
        self.srm_branches = nn.ModuleList()
        for name, num_filters, (start_idx, end_idx) in self.branch_configs:
            # Each branch: 3 RGB channels → (num_filters × 3) output channels
            srm_conv = nn.Conv2d(3, num_filters * 3, kernel_size=5, padding=2, groups=3, bias=False)
            with torch.no_grad():
                # Repeat the filter subset for each RGB channel
                srm_subset = srm_kernel[start_idx:end_idx]  # (num_filters, 5, 5)
                srm_subset_rgb = srm_subset.unsqueeze(1).repeat(3, 1, 1, 1)  # (3, num_filters, 5, 5) → (num_filters*3, 1, 5, 5)
                srm_conv.weight.copy_(srm_subset_rgb)
            self.srm_branches.append(srm_conv)

        # Create encoder branches (one ResidualStage per SRM branch)
        self.encoder_branches = nn.ModuleList()
        total_encoder_out_ch = 0
        for name, num_filters, _ in self.branch_configs:
            in_ch = num_filters * 3  # e.g., 8 filters × 3 RGB = 24 channels
            encoder = ResidualStage(in_ch=in_ch, maps_per_input=4, k=3, pool=4)
            self.encoder_branches.append(encoder)
            total_encoder_out_ch += encoder.out_ch

        # Stage2 and Stage3 (applied to concatenated encoder outputs)
        self.stage2 = ResidualStage(in_ch=total_encoder_out_ch, maps_per_input=4, k=3, pool=4)
        self.stage3 = DualAttentionStage(in_ch=self.stage2.out_ch, maps_per_input=2, k=3, pool=4)

        # Decoder: Reconstructs the 90-channel SRM-filtered image
        # Calculate optimal GroupNorm groups for decoder layers
        num_groups_stage2 = max(self.stage2.out_ch // 16, 1)
        while self.stage2.out_ch % num_groups_stage2 != 0:
            num_groups_stage2 -= 1

        num_groups_total_enc = max(total_encoder_out_ch // 16, 1)
        while total_encoder_out_ch % num_groups_total_enc != 0:
            num_groups_total_enc -= 1

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(self.stage3.out_ch, self.stage2.out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_stage2, num_channels=self.stage2.out_ch), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(self.stage2.out_ch, total_encoder_out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups_total_enc, num_channels=total_encoder_out_ch), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(total_encoder_out_ch, 90, kernel_size=3, padding=1, bias=True),
        )

        # ============================================
        # METHOD 2: LEARNABLE AMPLIFICATION NETWORK
        # ============================================
        # NO statistics tracking! Model learns amplification directly from data.
        # This solves the generalization problem of Method 1 (z-score based).

        # Learnable Amplification Network
        # Input: diff_map (90 channels) → Output: amplification factor (90 channels)
        self.amplification_net = nn.Sequential(
            # First convolution: reduce channels for efficiency
            nn.Conv2d(90, 45, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=9, num_channels=45),  # 45 / 9 = 5 channels per group
            nn.ReLU(inplace=True),

            # Second convolution: learn spatial patterns
            nn.Conv2d(45, 45, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=9, num_channels=45),
            nn.ReLU(inplace=True),

            # Third convolution: expand back to 90 channels
            nn.Conv2d(45, 90, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()  # Output range [0, 1]
        )

        # Maximum amplification factor (will be learned adaptively)
        self.max_amplification = 10.0  # Scale sigmoid output to [0, 10]

    def encode(self, x):
        """
        Multi-branch encoding:
        1. Apply SRM filter branches in parallel
        2. Encode each branch separately (Stage1 per branch)
        3. Concatenate and pass through Stage2, Stage3
        """
        # Apply SRM filter branches in parallel
        filtered_branches = [srm_branch(x) for srm_branch in self.srm_branches]

        # Encode each branch separately
        encoded_branches = [encoder(filtered) for encoder, filtered in zip(self.encoder_branches, filtered_branches)]

        # Concatenate all filtered outputs (for reconstruction target)
        x_filtered = torch.cat(filtered_branches, dim=1)  # (B, 90, H, W)

        # Concatenate all encoded outputs (for further processing)
        z_concat = torch.cat(encoded_branches, dim=1)  # (B, total_encoder_out_ch, H/4, W/4)

        # Pass through Stage2 and Stage3
        z = self.stage2(z_concat)
        z = self.stage3(z)

        return z, x_filtered

    def forward(self, x, is_cover=False):
        """
        Forward pass with METHOD 2: LEARNABLE AMPLIFICATION.

        Args:
            x: RGB input image (B, 3, H, W)
            is_cover: Unused (kept for API compatibility)

        Returns:
            x_hat: Reconstructed SRM-filtered image
            z: Latent encoding
            x_filtered: SRM-filtered input
            amplified_diff_map: Reconstruction error with LEARNED amplification
        """
        if x.numel() == 0:
            # Handle empty input case
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

        # 1. Encode input
        z, x_filtered = self.encode(x)

        # 2. Decode to reconstruct
        x_hat = self.decoder(z)

        # 3. Compute reconstruction error
        diff_map = torch.abs(x_filtered - x_hat)  # (B, 90, H, W)

        # ============================================
        # METHOD 2: LEARNABLE AMPLIFICATION
        # ============================================
        # Learn amplification factor directly from diff_map
        # NO statistics needed! Gradient flows end-to-end.

        # Amplification network predicts importance of each pixel
        # Output range: [0, 1] from Sigmoid
        learned_amplification = self.amplification_net(diff_map)  # (B, 90, H, W)

        # Scale to [0, max_amplification]
        # Areas with large errors OR stego artifacts → high amplification
        # Areas with small errors → low amplification
        learned_amplification = learned_amplification * self.max_amplification

        # Apply learned amplification
        amplified_diff_map = diff_map * learned_amplification

        # Return everything needed for loss calculation
        return x_hat, z, x_filtered, amplified_diff_map

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (NT-Xent) to push Cover and Stego embeddings apart.
    This version treats all cover images in a batch as positive to each other,
    and all stego images as positive to each other. Cover-stego pairs are negative.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_cover, z_stego):
        B = z_cover.shape[0]
        # Contrastive loss requires at least 2 samples of each type (cover/stego)
        # to form positive pairs. If B < 2, there are no pairs to compare.
        if B < 2:
            return torch.tensor(0.0, device=z_cover.device, requires_grad=True)

        # Normalize features and concatenate
        z_cover_norm = F.normalize(z_cover, dim=1)
        z_stego_norm = F.normalize(z_stego, dim=1)
        z_all = torch.cat([z_cover_norm, z_stego_norm], dim=0)

        # Calculate similarity matrix
        sim_matrix = torch.matmul(z_all, z_all.T) / self.temperature

        # Create the mask for positive pairs
        # The first B samples are covers, the next B are stegos.
        # All covers are positive to other covers.
        # All stegos are positive to other stegos.
        pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        pos_mask[:B, :B] = True  # Cover-Cover pairs
        pos_mask[B:, B:] = True  # Stego-Stego pairs

        # Mask out self-similarity from positive pairs
        pos_mask.fill_diagonal_(False)

        # We use log_softmax for numerical stability
        # Mask out self-similarity before softmax to prevent model from using it

        log_prob = F.log_softmax(sim_matrix, dim=1)

        # Sum log probabilities of positive pairs for each anchor
        # This is the numerator of the NT-Xent loss
        sum_log_prob_pos = (log_prob * pos_mask).sum(dim=1)

        # The number of positive pairs for each anchor
        num_positives = pos_mask.sum(dim=1)
        # Avoid division by zero for anchors with no positive pairs
        num_positives = num_positives.clamp(min=1)

        # Calculate the mean log probability for positive pairs
        mean_log_prob_pos = sum_log_prob_pos / num_positives

        # The loss is the negative of this mean, averaged over the batch
        loss = -mean_log_prob_pos.mean()

        return torch.clamp(loss, min=0.0, max=100.0)


class SNRLoss(nn.Module):
    """
    Signal-to-Noise Ratio (SNR) Loss for Steganalysis.

    Encourages the model to:
    1. Produce LOW amplified_diff for Cover images (clean reconstruction)
    2. Produce HIGH amplified_diff for Stego images (noisy reconstruction)

    This is batch-size independent and directly optimizes the detection goal.
    """
    def __init__(self, target_ratio=1.5, margin=0.2, loss_type='ratio', use_embedding_diff=False):
        """
        Args:
            target_ratio: Target ratio of stego_noise / cover_noise (default: 1.5)
            margin: Margin for margin-based loss (default: 0.2)
            loss_type: 'ratio' or 'margin' (default: 'ratio')
            use_embedding_diff: If True, directly compare embedding artifacts (stego-cover) instead of absolute noise levels (default: False)
        """
        super().__init__()
        self.target_ratio = target_ratio
        self.margin = margin
        self.loss_type = loss_type
        self.use_embedding_diff = use_embedding_diff

    def forward(self, cover_amplified_diff, stego_amplified_diff):
        """
        Args:
            cover_amplified_diff: (B, C, H, W) - Amplified reconstruction error for Cover
            stego_amplified_diff: (B, C, H, W) - Amplified reconstruction error for Stego

        Returns:
            loss: Scalar tensor
        """
        if self.use_embedding_diff:
            # NEW: Embedding-aware loss - directly isolate embedding artifacts
            # By computing (stego_diff - cover_diff), we remove natural image noise
            # and focus purely on embedding signal
            embedding_signal = stego_amplified_diff - cover_amplified_diff  # (B, C, H, W)

            # Measure magnitude of isolated embedding artifacts
            embedding_magnitude = embedding_signal.abs().mean(dim=[1, 2, 3])  # (B,)

            # Loss: Encourage embedding_magnitude to be LARGE (detectable)
            # If embedding is too small, loss increases
            loss = F.relu(self.margin - embedding_magnitude).mean()

        else:
            # ORIGINAL: Compare absolute noise levels
            cover_noise = cover_amplified_diff.mean(dim=[1, 2, 3])  # (B,)
            stego_noise = stego_amplified_diff.mean(dim=[1, 2, 3])  # (B,)

            if self.loss_type == 'ratio':
                # Ratio-based: Encourage stego_noise / cover_noise >= target_ratio
                # Loss = max(0, target_ratio - (stego / cover))
                ratio = stego_noise / (cover_noise + 1e-6)
                loss = F.relu(self.target_ratio - ratio).mean()

            elif self.loss_type == 'margin':
                # Margin-based: Encourage stego_noise - cover_noise >= margin
                # Loss = max(0, margin - (stego - cover))
                diff = stego_noise - cover_noise
                loss = F.relu(self.margin - diff).mean()

            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss

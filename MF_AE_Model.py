import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchvision import models

# -------------------------
# 다중 스케일 이미지 필터링 모듈
# -------------------------
class MultiScaleImageFilters(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 다중 스케일 LoG 필터 (엣지 탐지)
        self.register_buffer('log_kernel_3x3', self._create_log_kernel(3, sigma=0.8))   # 미세 엣지
        self.register_buffer('log_kernel_5x5', self._create_log_kernel(5, sigma=1.0))   # 중간 엣지
        self.register_buffer('log_kernel_7x7', self._create_log_kernel(7, sigma=1.4))   # 큰 엣지
        
        # 다중 스케일 HPF 필터 (디테일 탐지)
        self.register_buffer('hpf_kernel_3x3', self._create_hpf_kernel(3))
        self.register_buffer('hpf_kernel_5x5', self._create_hpf_kernel(5))
        
        # LPF 필터 (구조 보존)
        self.register_buffer('lpf_kernel', self._create_lpf_kernel())
        
        # Gabor 필터들 (방향성 텍스처)
        for angle in [0, 45, 90, 135]:
            self.register_buffer(f'gabor_{angle}', self._create_gabor_kernel(angle))
        
    def _create_log_kernel(self, kernel_size, sigma=1.0):
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        laplacian = -1 / (np.pi * sigma**4) * (1 - (xx**2 + yy**2) / (2 * sigma**2))
        log_kernel = laplacian * gaussian
        log_kernel = log_kernel / torch.sum(torch.abs(log_kernel))
        
        return log_kernel.unsqueeze(0).unsqueeze(0)
    
    def _create_hpf_kernel(self, kernel_size):
        if kernel_size == 3:
            hpf = torch.tensor([
                [-1, -1, -1],
                [-1,  8, -1], 
                [-1, -1, -1]
            ], dtype=torch.float32)
        elif kernel_size == 5:
            hpf = torch.tensor([
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, 24, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1]
            ], dtype=torch.float32)
        
        return hpf.unsqueeze(0).unsqueeze(0)
    
    def _create_lpf_kernel(self):
        lpf = torch.tensor([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=torch.float32) / 16.0
        
        return lpf.unsqueeze(0).unsqueeze(0)
    
    def _create_gabor_kernel(self, angle, frequency=0.1, sigma_x=2, sigma_y=2):
        kernel_size = 7
        theta = np.radians(angle)
        
        x, y = np.meshgrid(np.linspace(-3, 3, kernel_size), 
                          np.linspace(-3, 3, kernel_size))
        
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        gabor = np.exp(-(x_theta**2/sigma_x**2 + y_theta**2/sigma_y**2)/2) * \
                np.cos(2 * np.pi * frequency * x_theta)
        
        gabor_tensor = torch.from_numpy(gabor).float()
        gabor_tensor = gabor_tensor / torch.sum(torch.abs(gabor_tensor))
        
        return gabor_tensor.unsqueeze(0).unsqueeze(0)
    
    def _apply_filter_to_channels(self, x, kernel, padding):
        B, C, H, W = x.shape
        filtered_channels = []
        
        for c in range(C):
            channel = x[:, c:c+1]
            filtered = F.conv2d(channel, kernel, padding=padding)
            filtered_channels.append(filtered)
        
        return torch.cat(filtered_channels, dim=1)
    
    def forward(self, x, use_gabor=False):
        B, C, H, W = x.shape
        results = {}
        
        # 원본
        results['original'] = x
        
        # LPF (구조 정보)
        lpf_response = self._apply_filter_to_channels(x, self.lpf_kernel, padding=1)
        results['lpf'] = lpf_response
        
        # 다중 스케일 LoG (엣지 정보)
        log_3 = self._apply_filter_to_channels(x, self.log_kernel_3x3, padding=1)
        log_5 = self._apply_filter_to_channels(x, self.log_kernel_5x5, padding=2)
        log_7 = self._apply_filter_to_channels(x, self.log_kernel_7x7, padding=3)
        
        # 다중 스케일 LoG 융합
        log_combined = torch.stack([
            (torch.tanh(log_3 * 0.1) + 1) / 2,
            (torch.tanh(log_5 * 0.1) + 1) / 2,
            (torch.tanh(log_7 * 0.1) + 1) / 2
        ], dim=1)
        results['log'] = log_combined.max(dim=1)[0]  # 최대값으로 융합
        
        # 다중 스케일 HPF (디테일 정보)
        hpf_3 = self._apply_filter_to_channels(x, self.hpf_kernel_3x3, padding=1)
        hpf_5 = self._apply_filter_to_channels(x, self.hpf_kernel_5x5, padding=2)
        
        hpf_combined = torch.stack([
            (torch.tanh(hpf_3 * 0.1) + 1) / 2,
            (torch.tanh(hpf_5 * 0.1) + 1) / 2
        ], dim=1)
        results['hpf'] = hpf_combined.max(dim=1)[0]
        
        # Gabor 필터 (방향성 텍스처)
        if use_gabor:
            gabor_responses = []
            for angle in [0, 45, 90, 135]:
                gabor_kernel = getattr(self, f'gabor_{angle}')
                gabor_response = self._apply_filter_to_channels(x, gabor_kernel, padding=3)
                gabor_responses.append(torch.tanh(gabor_response * 0.1))
            
            gabor_combined = torch.stack(gabor_responses, dim=1)
            results['gabor'] = gabor_combined.max(dim=1)[0]
        
        return results

# -------------------------
# 어텐션 가이드 융합 모듈
# -------------------------
class AttentionGuidedFusion(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        
        # Option 1: Attention Map 생성기들
        self.edge_attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim//4, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim//4, 1, 1),
            nn.Sigmoid()
        )
        
        self.texture_attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim//4, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim//4, 1, 1),
            nn.Sigmoid()
        )
        
        self.structure_attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim//4, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim//4, 1, 1),
            nn.Sigmoid()
        )
        
        # Option 2: Multi-Scale Guidance
        self.multiscale_guidance = nn.ModuleList([
            nn.Conv2d(feature_dim, 1, 1),  # Fine scale
            nn.Conv2d(feature_dim, 1, 1),  # Medium scale  
            nn.Conv2d(feature_dim, 1, 1)   # Coarse scale
        ])
        
        # Original 브랜치 강화
        self.original_enhancer = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # 가이던스 융합
        self.guidance_fusion = nn.Conv2d(6, 1, 1)  # 6 = 3(attention) + 3(multiscale)
        
    def forward(self, branch_features):
        original_feat, lpf_feat, log_feat, hpf_feat = branch_features[:4]
        
        # Option 1: 각 필터에서 어텐션 맵 추출
        edge_attn = self.edge_attention(log_feat)        # 엣지 중요도
        texture_attn = self.texture_attention(hpf_feat)  # 텍스처 중요도
        structure_attn = self.structure_attention(lpf_feat) # 구조 중요도
        
        # Option 2: Multi-Scale Guidance
        # 다운샘플링으로 다양한 스케일 생성
        scales = [original_feat]
        for i in range(2):
            downsampled = F.avg_pool2d(scales[-1], 2)
            scales.append(downsampled)
        
        # 각 스케일에서 가이던스 추출
        multiscale_guides = []
        for i, (scale_feat, guide_conv) in enumerate(zip(scales, self.multiscale_guidance)):
            guide = torch.sigmoid(guide_conv(scale_feat))
            # 원래 크기로 업샘플링
            if i > 0:
                guide = F.interpolate(guide, size=original_feat.shape[2:], mode='bilinear')
            multiscale_guides.append(guide)
        
        # 모든 가이던스 결합
        all_guidance = torch.cat([
            edge_attn, texture_attn, structure_attn,
            *multiscale_guides
        ], dim=1)
        
        # 통합 어텐션 맵 생성
        unified_attention = torch.sigmoid(self.guidance_fusion(all_guidance))
        
        # Original 피처 강화
        enhanced_original = self.original_enhancer(original_feat)
        
        # 어텐션 가이드 적용
        guided_original = enhanced_original * (1 + 2.0 * unified_attention)
        
        return guided_original, {
            'edge_attention': edge_attn,
            'texture_attention': texture_attn,
            'structure_attention': structure_attn,
            'multiscale_guidance': multiscale_guides,
            'unified_attention': unified_attention
        }

# -------------------------
# ResNet 기반 브랜치 인코더
# -------------------------
class BranchEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, pretrained=True):
        super().__init__()
        
        resnet18 = models.resnet18(pretrained=pretrained)
        
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        resnet_out_channels = 512
        if out_channels != resnet_out_channels:
            self.output_adapter = nn.Sequential(
                nn.Conv2d(resnet_out_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.output_adapter = nn.Identity()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.output_adapter(x)
        return x

# -------------------------
# MemAE 구성요소들
# -------------------------
def hard_shrink_relu(input, lambd=0, epsilon=1e-6):
    input = torch.clamp(input, min=-10, max=10)
    denominator = torch.abs(input - lambd) + epsilon
    denominator = torch.clamp(denominator, min=epsilon)
    output = (F.relu(input - lambd) * input) / denominator
    output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
    output = torch.where(torch.isinf(output), torch.zeros_like(output), output)
    return output

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, temperature=0.5):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input):
        input = torch.clamp(input, min=-10, max=10)
        
        att_weight = F.linear(input, self.weight)
        att_weight = F.softmax(att_weight / self.temperature, dim=1)
        
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            weight_sum = att_weight.sum(dim=1, keepdim=True)
            zero_mask = (weight_sum < 1e-8)
            att_weight = att_weight / (weight_sum + 1e-8)
            att_weight[zero_mask.squeeze()] = 1.0 / self.mem_dim
            
        att_weight = torch.where(torch.isnan(att_weight), 
                                torch.ones_like(att_weight) / self.mem_dim, 
                                att_weight)
        att_weight = torch.where(torch.isinf(att_weight), 
                                torch.ones_like(att_weight) / self.mem_dim, 
                                att_weight)
            
        mem_trans = self.weight.permute(1, 0)
        output = F.linear(att_weight, mem_trans)
        
        return {'output': output, 'att': att_weight}

class MemAEMemoryNetwork(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemAEMemoryNetwork, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)
        
    def forward(self, input, update_indices=None):
        B, N, C = input.shape
        x = input.view(-1, C)
        memory_output = self.memory(x)
        output = memory_output['output'].view(B, N, C)
        return output

# -------------------------
# 개선된 디코더
# -------------------------
class SimpleDecoder(nn.Module):
    def __init__(self, feature_dim=256, input_size=7):
        super().__init__()
        
        self.decoder = nn.Sequential(
            # 7x7 → 14x14
            nn.ConvTranspose2d(feature_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 14x14 → 28x28  
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 28x28 → 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 56x56 → 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 112x112 → 224x224
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(x)

# -------------------------
# 메인 모델
# -------------------------
class MultiFilterMemAE(nn.Module):
    def __init__(self, feature_dim=256, mem_dim=512, patch_size=7, shrink_thres=0.0001, use_gabor=False):
        super().__init__()
        
        self.patch_size = patch_size
        self.use_gabor = use_gabor
        
        # 다중 스케일 이미지 필터링
        self.filters = MultiScaleImageFilters()
        
        # 브랜치별 인코더
        self.original_encoder = BranchEncoder(3, feature_dim)
        self.lpf_encoder = BranchEncoder(3, feature_dim)
        self.log_encoder = BranchEncoder(3, feature_dim)
        self.hpf_encoder = BranchEncoder(3, feature_dim)
        
        if use_gabor:
            self.gabor_encoder = BranchEncoder(3, feature_dim)
        
        # 어텐션 가이드 융합
        self.attention_fusion = AttentionGuidedFusion(feature_dim)
        
        # MemAE 메모리
        patch_dim = feature_dim * patch_size * patch_size
        self.memory_net = MemAEMemoryNetwork(mem_dim, patch_dim, shrink_thres)
        
        # 디코더
        self.decoder = SimpleDecoder(feature_dim)
        
    def forward(self, x, compute_loss=False, return_details=False):
        B, C, H, W = x.shape
        
        # 1. 다중 스케일 이미지 필터링
        filtered_results = self.filters(x, use_gabor=self.use_gabor)
        
        # 2. 각 브랜치에서 특징 추출
        original_feat = self.original_encoder(filtered_results['original'])
        lpf_feat = self.lpf_encoder(filtered_results['lpf'])
        log_feat = self.log_encoder(filtered_results['log'])
        hpf_feat = self.hpf_encoder(filtered_results['hpf'])
        
        branch_features = [original_feat, lpf_feat, log_feat, hpf_feat]
        
        if self.use_gabor and 'gabor' in filtered_results:
            gabor_feat = self.gabor_encoder(filtered_results['gabor'])
            branch_features.append(gabor_feat)
        
        # 3. 어텐션 가이드 융합
        guided_feat, attention_maps = self.attention_fusion(branch_features)
        
        # 4. 패치 기반 메모리 처리
        H_feat, W_feat = guided_feat.shape[2:]
        patch = self.patch_size
        
        patches = guided_feat.unfold(2, patch, patch).unfold(3, patch, patch)
        patches = patches.contiguous().view(B, guided_feat.shape[1], -1, patch, patch)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        num_patches = patches.shape[1]
        patches_flat = patches.view(B, num_patches, -1)
        
        try:
            retrieved_patches_flat = self.memory_net(patches_flat)
            retrieved_flat = retrieved_patches_flat.transpose(1, 2).contiguous()
            retrieved_feat = F.fold(
                retrieved_flat,
                output_size=(H_feat, W_feat),
                kernel_size=patch,
                stride=patch
            )
        except Exception as e:
            retrieved_feat = guided_feat
        
        reconstructed = self.decoder(retrieved_feat)
        
        if return_details:
            return {
                'reconstructed': reconstructed,
                'filtered_results': filtered_results,
                'attention_maps': attention_maps,
                'branch_features': [f.detach() for f in branch_features],
                'guided_feat': guided_feat,
            }
        
        if compute_loss:
            # Option 3: Loss-Level Guidance
            edge_mask = (filtered_results['log'] > 0.6).float()
            detail_mask = (filtered_results['hpf'] > 0.6).float()
            
            # 기본 loss
            mse_loss = F.mse_loss(reconstructed, x)
            
            # 가이드된 loss (중요 영역 강조)
            edge_loss = F.mse_loss(reconstructed * edge_mask, x * edge_mask) * 2.0
            detail_loss = F.mse_loss(reconstructed * detail_mask, x * detail_mask) * 1.5
            
            # SSIM loss
            try:
                from pytorch_msssim import ssim
                ssim_value = ssim(reconstructed, x, data_range=1.0)
                ssim_loss = 1 - ssim_value
            except ImportError:
                ssim_loss = torch.tensor(0.0, device=x.device)
            
            # 어텐션 정규화 (너무 집중되지 않도록)
            attention_entropy = -torch.sum(
                attention_maps['unified_attention'] * 
                torch.log(attention_maps['unified_attention'] + 1e-8)
            ) / (B * H_feat * W_feat)
            
            # 전체 loss
            total_loss = (
                mse_loss + 
                edge_loss + 
                detail_loss +
                ssim_loss - 
                0.01 * attention_entropy  # 어텐션 다양성 장려
            )
            
            return total_loss, {
                'mse_loss': mse_loss.item(),
                'edge_loss': edge_loss.item(),
                'detail_loss': detail_loss.item(),
                'ssim_loss': ssim_loss.item() if isinstance(ssim_loss, torch.Tensor) else 0.0,
                'attention_entropy': attention_entropy.item(),
                'total_loss': total_loss.item()
            }
        
        return reconstructed
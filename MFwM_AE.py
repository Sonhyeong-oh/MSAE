import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

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
    
    def _apply_filter_to_channels(self, x, kernel, padding):
        B, C, H, W = x.shape
        filtered_channels = []
        
        for c in range(C):
            channel = x[:, c:c+1]
            filtered = F.conv2d(channel, kernel, padding=padding)
            filtered_channels.append(filtered)
        
        return torch.cat(filtered_channels, dim=1)
    
    def forward(self, x):
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
        
        return results

# -------------------------
# 필터별 특화 CNN 인코더
# -------------------------
class SpecializedCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, encoder_type='original'):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'original':
            # 원본: 전반적인 특징을 균형있게 추출 (표준 CNN)
            self.encoder = self._build_standard_encoder(in_channels, out_channels)
        elif encoder_type == 'lpf':
            # LPF: 저주파/구조적 특징에 특화 (더 큰 receptive field)
            self.encoder = self._build_structure_encoder(in_channels, out_channels)
        elif encoder_type == 'log':
            # LoG: 엣지 특징에 특화 (더 세밀한 필터)
            self.encoder = self._build_edge_encoder(in_channels, out_channels)
        elif encoder_type == 'hpf':
            # HPF: 텍스처/디테일에 특화 (얕은 네트워크, 미세한 패턴 중심)
            self.encoder = self._build_texture_encoder(in_channels, out_channels)
    
    def _build_standard_encoder(self, in_channels, out_channels):
        """표준 CNN 인코더 - 전반적인 특징 추출"""
        return nn.Sequential(
            # Layer 1: 224x224 → 112x112
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 112x112 → 56x56
            
            # Layer 2: 56x56 → 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 3: 28x28 → 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4: 14x14 → 7x7
            nn.Conv2d(256, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _build_structure_encoder(self, in_channels, out_channels):
        """구조 특화 인코더 - 큰 receptive field와 깊은 구조"""
        return nn.Sequential(
            # Layer 1: 큰 커널로 시작 (전역 구조 파악)
            nn.Conv2d(in_channels, 32, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Layer 2: 큰 receptive field 유지
            nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 3: 구조적 패턴 강화
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 4: 최종 구조 특징
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Adaptive pooling + final projection
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(256, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _build_edge_encoder(self, in_channels, out_channels):
        """엣지 특화 인코더 - 세밀한 엣지 탐지"""
        return nn.Sequential(
            # Layer 1: 작은 커널로 시작 (미세한 엣지)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 엣지 강화를 위한 dilated convolution
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224x224 → 112x112
            
            # Layer 2: 다중 스케일 엣지
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 112x112 → 56x56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Parallel edge detection (3x3, 5x5)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56 → 28x28
            
            # Layer 3: 엣지 통합
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 28x28 → 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4: 최종 엣지 특징
            nn.Conv2d(256, out_channels, kernel_size=3, stride=2, padding=1),  # 14x14 → 7x7
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _build_texture_encoder(self, in_channels, out_channels):
        """텍스처 특화 인코더 - 국소적 패턴에 집중"""
        return nn.Sequential(
            # Layer 1: 미세한 텍스처 포착 (작은 stride)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224x224 → 112x112
            
            # Layer 2: 텍스처 패턴 감지
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 다양한 크기의 텍스처를 위한 병렬 처리
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),  # 112x112 → 56x56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 3: 텍스처 통합
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 56x56 → 28x28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4: 고주파 디테일 보존
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 28x28 → 14x14
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Layer 5: 최종 텍스처 특징 (더 깊은 네트워크)
            nn.Conv2d(512, out_channels, kernel_size=3, stride=2, padding=1),  # 14x14 → 7x7
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.encoder(x)

# -------------------------
# Adaptive Soft Loss 계산기
# -------------------------
class AdaptiveSoftLossCalculator:
    def __init__(self, edge_percentile=0.8, detail_percentile=0.7, structure_percentile=0.6):
        self.edge_percentile = edge_percentile
        self.detail_percentile = detail_percentile
        self.structure_percentile = structure_percentile
    
    def _compute_adaptive_mask(self, response, percentile, sharpness=5.0):
        B, C, H, W = response.shape
        
        # 배치별로 동적 임계값 계산
        response_flat = response.view(B, -1)
        adaptive_thresholds = torch.quantile(response_flat, percentile, dim=1, keepdim=True)
        adaptive_thresholds = adaptive_thresholds.view(B, 1, 1, 1)
        
        # 적응적 정규화
        normalized_response = response / (adaptive_thresholds + 1e-8)
        
        # 부드러운 마스크 생성
        soft_mask = torch.sigmoid(sharpness * (normalized_response - 1.0))
        
        return soft_mask
    
    def compute_adaptive_edge_loss(self, reconstructed, original, log_response, weight=2.0):
        edge_mask = self._compute_adaptive_mask(log_response, self.edge_percentile, sharpness=6.0)
        pixel_losses = (reconstructed - original) ** 2
        weighted_losses = pixel_losses * edge_mask
        return weighted_losses.mean() * weight
    
    def compute_adaptive_detail_loss(self, reconstructed, original, hpf_response, weight=1.5):
        detail_mask = self._compute_adaptive_mask(hpf_response, self.detail_percentile, sharpness=5.0)
        pixel_losses = (reconstructed - original) ** 2
        weighted_losses = pixel_losses * detail_mask
        return weighted_losses.mean() * weight

class MultiFilterAdaptiveLoss:
    def __init__(self, edge_weight=2.0, detail_weight=1.5, base_mse_weight=1.0):
        self.calculator = AdaptiveSoftLossCalculator()
        self.edge_weight = edge_weight
        self.detail_weight = detail_weight
        self.base_mse_weight = base_mse_weight
    
    def compute_total_loss(self, reconstructed, original, filtered_results, attention_maps):
        loss_dict = {}
        
        # 기본 MSE
        base_mse = F.mse_loss(reconstructed, original) * self.base_mse_weight
        loss_dict['base_mse'] = base_mse.item()
        
        # 적응적 엣지 손실
        edge_loss = self.calculator.compute_adaptive_edge_loss(
            reconstructed, original, filtered_results['log'], self.edge_weight
        )
        loss_dict['adaptive_edge'] = edge_loss.item()
        
        # 적응적 디테일 손실
        detail_loss = self.calculator.compute_adaptive_detail_loss(
            reconstructed, original, filtered_results['hpf'], self.detail_weight
        )
        loss_dict['adaptive_detail'] = detail_loss.item()
        
        total_loss = base_mse + edge_loss + detail_loss
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict

# -------------------------
# 어텐션 관련 모듈들
# -------------------------
class SimpleAttentionGenerator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
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

    def forward(self, branch_features):
        original_feat, lpf_feat, log_feat, hpf_feat = branch_features
        edge_attn = self.edge_attention(log_feat)
        texture_attn = self.texture_attention(hpf_feat)
        return [edge_attn, texture_attn]

class AttentionGuidedFusion(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        
        self.attention_generator = SimpleAttentionGenerator(feature_dim)
        
        # Original 브랜치 강화
        self.original_enhancer = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # 어텐션 융합
        self.guidance_fusion = nn.Conv2d(2, 1, 1)  # edge + texture attention
        
    def forward(self, branch_features):
        original_feat = branch_features[0]
        
        # 어텐션 생성
        attentions = self.attention_generator(branch_features)
        attention_maps = {
            'edge_attention': attentions[0],
            'texture_attention': attentions[1]
        }
        
        # 통합 어텐션
        all_guidance_tensor = torch.cat(attentions, dim=1)
        unified_attention = torch.sigmoid(self.guidance_fusion(all_guidance_tensor))
        attention_maps['unified_attention'] = unified_attention
        
        # 피처 강화
        enhanced_original = self.original_enhancer(original_feat)
        guided_original = enhanced_original * (1 + 2.0 * unified_attention)
        
        return guided_original, attention_maps

# -------------------------
# 디코더
# -------------------------
class SimpleDecoder(nn.Module):
    def __init__(self, feature_dim=256):
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
class MultiFilterAutoEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        
        # 다중 스케일 이미지 필터링
        self.filters = MultiScaleImageFilters()
        self.loss_calculator = MultiFilterAdaptiveLoss()
        
        # 필터별 특화 인코더
        self.original_encoder = SpecializedCNNEncoder(3, feature_dim, encoder_type='original')
        self.lpf_encoder = SpecializedCNNEncoder(3, feature_dim, encoder_type='lpf')
        self.log_encoder = SpecializedCNNEncoder(3, feature_dim, encoder_type='log')
        self.hpf_encoder = SpecializedCNNEncoder(3, feature_dim, encoder_type='hpf')
        
        # 어텐션 가이드 융합
        self.attention_fusion = AttentionGuidedFusion(feature_dim)
        
        # 디코더
        self.decoder = SimpleDecoder(feature_dim)
        
    def compute_specialization_loss(self, branch_features, filtered_results, original):
        """각 브랜치가 해당 필터의 특징을 잘 학습하도록 하는 특화 손실"""
        spec_losses = {}
        
        # Original 브랜치: 전반적인 재구성 능력
        original_recon = self.decoder(branch_features[0])
        original_loss = F.mse_loss(original_recon, original)
        spec_losses['original_spec'] = original_loss.item()
        
        # LoG 브랜치: 엣지 중심 재구성 능력
        log_recon = self.decoder(branch_features[2])
        edge_importance = self._compute_edge_importance(filtered_results['log'])
        log_loss = F.mse_loss(log_recon * edge_importance, original * edge_importance) * 1.5
        spec_losses['log_spec'] = log_loss.item()
        
        # HPF 브랜치: 텍스처 재구성 능력
        hpf_recon = self.decoder(branch_features[3])
        texture_importance = self._compute_texture_importance(filtered_results['hpf'])
        hpf_loss = F.mse_loss(hpf_recon * texture_importance, original * texture_importance) * 1.3
        spec_losses['hpf_spec'] = hpf_loss.item()
        
        total_spec_loss = original_loss + log_loss + hpf_loss
        spec_losses['total_specialization'] = total_spec_loss.item()
        
        return total_spec_loss, spec_losses
    
    def _compute_edge_importance(self, log_response):
        B = log_response.shape[0]
        log_flat = log_response.view(B, -1)
        thresholds = torch.quantile(log_flat, 0.7, dim=1, keepdim=True).view(B, 1, 1, 1)
        normalized = log_response / (thresholds + 1e-8)
        importance = torch.sigmoid(3.0 * (normalized - 1.0))
        return importance
    
    def _compute_texture_importance(self, hpf_response):
        B = hpf_response.shape[0]
        hpf_flat = hpf_response.view(B, -1)
        thresholds = torch.quantile(hpf_flat, 0.6, dim=1, keepdim=True).view(B, 1, 1, 1)
        normalized = hpf_response / (thresholds + 1e-8)
        importance = torch.sigmoid(3.0 * (normalized - 1.0))
        return importance
        
    def forward(self, x, compute_loss=False, return_details=False, include_specialization_loss=True):
        B, C, H, W = x.shape
        
        # 1. 다중 스케일 이미지 필터링 및 특징 추출
        filtered_results = self.filters(x)
        original_feat = self.original_encoder(filtered_results['original'])
        lpf_feat = self.lpf_encoder(filtered_results['lpf'])
        log_feat = self.log_encoder(filtered_results['log'])
        hpf_feat = self.hpf_encoder(filtered_results['hpf'])
        branch_features = [original_feat, lpf_feat, log_feat, hpf_feat]
        
        # 2. 어텐션 가이드 융합
        guided_feat, attention_maps = self.attention_fusion(branch_features)
        
        # 3. 디코딩
        reconstructed = self.decoder(guided_feat)
        
        if return_details:
            return {
                'reconstructed': reconstructed,
                'filtered_results': filtered_results,
                'attention_maps': attention_maps,
                'branch_features': [f.detach() for f in branch_features],
                'guided_feat': guided_feat,
            }
        
        if compute_loss:
            # 기본 적응적 손실
            total_loss, loss_dict = self.loss_calculator.compute_total_loss(
                reconstructed, x, filtered_results, attention_maps
            )
            
            # 브랜치 특화 손실 추가
            if include_specialization_loss:
                spec_loss, spec_loss_dict = self.compute_specialization_loss(
                    branch_features, filtered_results, x
                )
                
                total_loss = total_loss + 0.3 * spec_loss
                loss_dict.update(spec_loss_dict)
                loss_dict['total'] = total_loss.item()
            
            return total_loss, loss_dict
        
        return reconstructed
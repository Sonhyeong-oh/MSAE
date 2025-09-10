from __future__ import absolute_import, print_function
import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import timm


# ------------------------------
# 1) MemoryUnit / MemModule (원형 유지, 약간 주석 보강)
# ------------------------------
class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        """
        mem_dim: 메모리 슬롯 수 (M)
        fea_dim: 피처 차원 (C)
        """
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        input: (T x C) 형태. 여기서 T=공간 위치 수(N*H*W 등), C=채널
        리턴:
          - output: (T x C)
          - att:    (T x M)  (메모리 슬롯에 대한 어텐션)
        """
        # (T x C) * (C x M) = (T x M)
        att_weight = F.linear(input, self.weight)
        att_weight = F.softmax(att_weight, dim=1)  # normalize over M

        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)  # L1 normalize

        mem_trans = self.weight.permute(1, 0)  # (C x M)
        # (T x M) * (M x C) = (T x C)
        output = F.linear(att_weight, mem_trans)
        return {'output': output, 'att': att_weight}

    def extra_repr(self):
        return f'mem_dim={self.mem_dim}, fea_dim={self.fea_dim}'


class MemModule(nn.Module):
    """
    입력: N x C x H x W (또는 3D/5D도 지원)
    내부에서 (N*H*W) x C 로 펼쳐 MemoryUnit 통과 → 다시 원래 텐서 형태로 복원
    """
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)

        if l == 3:      # (N, C, T)
            x = input.permute(0, 2, 1)
        elif l == 4:    # (N, C, H, W)
            x = input.permute(0, 2, 3, 1)
        elif l == 5:    # (N, C, D, H, W)
            x = input.permute(0, 2, 3, 4, 1)
        else:
            raise ValueError('wrong feature map size')

        x = x.contiguous().view(-1, s[1])  # (T, C)
        y_and = self.memory(x)
        y, att = y_and['output'], y_and['att']  # (T, C), (T, M)

        if l == 3:
            y = y.view(s[0], s[2], s[1]).permute(0, 2, 1)                 # (N, C, T)
            att = att.view(s[0], s[2], self.mem_dim).permute(0, 2, 1)     # (N, M, T)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1]).permute(0, 3, 1, 2)        # (N, C, H, W)
            att = att.view(s[0], s[2], s[3], self.mem_dim).permute(0, 3, 1, 2)  # (N, M, H, W)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1]).permute(0, 4, 1, 2, 3)     # (N, C, D, H, W)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim).permute(0, 4, 1, 2, 3)  # (N, M, D, H, W)

        return {'output': y, 'att': att}


def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    # 양수에 대해서만 hard-shrink 유사 동작
    return (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)


# ------------------------------
# 2) HRNet + Memory Attention + Feature Saving
# ------------------------------
class HRNetMemoryExtractor(nn.Module):
    """
    - HRNet(features_only=True)로 마지막 feature 추출
    - MemModule로 메모리-어텐션 적용
    - 메모리 어텐션을 통합해 중요도 맵(importance map) 생성
    - top-k 비율 또는 threshold로 중요 영역만 마스킹 → 저장 벡터 산출
    """
    def __init__(
        self,
        hrnet_variant: str = 'hrnet_w18',
        mem_dim: int = 256,
        shrink_thres: float = 0.0025,
        importance_reduce: str = 'max',  # 'max' | 'mean' (메모리 차원 축약 방식)
        use_proj_1x1: bool = False      # HRNet 채널을 메모리 fea_dim에 맞추고 싶을 때 사용
    ):
        super().__init__()
        # 1) HRNet Backbone
        self.hrnet = timm.create_model(
            hrnet_variant,
            pretrained=True,
            features_only=True
        )
        in_channels = self.hrnet.feature_info[-1]['num_chs']  # HRNet 최종 채널 수(C)

        # 2) (선택) 채널 정합 1x1 conv (fea_dim 변경 필요시)
        self.fea_dim = in_channels
        self.proj = None
        if use_proj_1x1:
            self.fea_dim = min(in_channels, 512)  # 예시: 512로 줄임
            self.proj = nn.Conv2d(in_channels, self.fea_dim, kernel_size=1, bias=False)

        # 3) Memory Attention
        self.mem = MemModule(mem_dim=mem_dim, fea_dim=self.fea_dim, shrink_thres=shrink_thres)
        self.mem_dim = mem_dim
        self.importance_reduce = importance_reduce

        # 4) 저장용 pooling (중요 영역만 평균 풀링해서 벡터화)
        self.save_pool = nn.AdaptiveAvgPool2d((1, 1))

    @torch.no_grad()
    def _compute_importance(self, att: torch.Tensor) -> torch.Tensor:
        """
        att: (N, M, H, W)  -> 중요도 맵 (N, 1, H, W)
        메모리 슬롯 차원(M) 축약
        """
        if self.importance_reduce == 'max':
            imp = att.max(dim=1, keepdim=True).values
        else:
            imp = att.mean(dim=1, keepdim=True)
        # 0~1 정규화
        imp = (imp - imp.amin(dim=(2,3), keepdim=True)) / (imp.amax(dim=(2,3), keepdim=True) - imp.amin(dim=(2,3), keepdim=True) + 1e-6)
        return imp

    @torch.no_grad()
    def _make_mask(self, importance: torch.Tensor, topk_ratio: float = 0.1, threshold: float = None):
        """
        importance: (N,1,H,W)
        topk_ratio: 상위 비율 (0~1), threshold가 주어지면 무시됨
        return: bool mask (N,1,H,W)
        """
        N, _, H, W = importance.shape
        if threshold is not None:
            mask = (importance >= threshold)
            return mask

        # top-k 비율 선택
        T = H * W
        k = max(1, int(T * topk_ratio))
        imp_flat = importance.view(N, -1)
        kth = torch.topk(imp_flat, k, dim=1).values[:, -1]  # 각 배치별 k번째 값
        kth = kth.view(N, 1, 1, 1)
        mask = (importance >= kth)
        return mask

    def forward(self, x: torch.Tensor, *, topk_ratio: float = 0.1, threshold: float = None, detach_for_saving: bool = True):
        """
        x: 입력 이미지 (N,3,H,W)
        반환 dict:
          - 'feat': HRNet 최종 feature (N,C,H',W')
          - 'mem_out': 메모리 출력 feature (N,C,H',W')
          - 'att_map': 메모리 어텐션 맵 (N,M,H',W')
          - 'importance': 중요도 맵 (N,1,H',W')
          - 'mask': 중요 영역 마스크(bool) (N,1,H',W')
          - 'masked_feat': 중요 영역만 남긴 feature (N,C,H',W')
          - 'saved_vec': 저장용 1D 벡터 (N,C)
        """
        # 1) HRNet feature
        feats = self.hrnet(x)
        feat = feats[-1]  # (N, C, H', W')

        if self.proj is not None:
            feat = self.proj(feat)  # (N, fea_dim, H', W')

        # 2) Memory attention
        out = self.mem(feat)
        mem_out = out['output']         # (N, C, H', W')
        att_map = out['att']            # (N, M, H', W')

        # 3) 중요도 맵 & 마스크
        importance = self._compute_importance(att_map)                 # (N,1,H',W')
        mask = self._make_mask(importance, topk_ratio=topk_ratio, threshold=threshold)  # bool

        # 4) 중요 영역 feature만 보존 (브로드캐스트)
        masked_feat = mem_out * mask

        # 5) 저장용 벡터 (중요 영역만 평균 풀링 → (N,C))
        # 중요 영역이 0일 경우 대비하여 작은 eps로 나눔
        if mask.dtype != mem_out.dtype:
            mask_f = mask.float()
        else:
            mask_f = mask

        # 가중 평균 풀링 (마스크된 영역만)
        denom = mask_f.sum(dim=(2,3), keepdim=True).clamp_min(1.0)     # (N,1,1,1)
        saved_vec = (mem_out * mask_f).sum(dim=(2,3), keepdim=True) / denom  # (N,C,1,1)
        saved_vec = saved_vec.view(saved_vec.size(0), -1)               # (N,C)

        if detach_for_saving:
            saved_vec = saved_vec.detach()

        return {
            'feat': feat,
            'mem_out': mem_out,
            'att_map': att_map,
            'importance': importance,
            'mask': mask,
            'masked_feat': masked_feat,
            'saved_vec': saved_vec
        }

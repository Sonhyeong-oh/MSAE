# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class ViT_AEpp(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 use_perc=False, use_edge=False, lambda_perc=0.01, lambda_edge=10.0,
                 use_contrastive=False, proj_hidden_dim=None):
        
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # === Perceptual loss(VGG) & Edge loss(Sobel) 추가 ===
        self.norm_pix_loss = norm_pix_loss
        self.use_perc = use_perc
        self.use_edge = use_edge
        self.lambda_perc = lambda_perc
        self.lambda_edge = lambda_edge
        self.use_contrastive = use_contrastive
        # encoder embed dim (= CLS dim)
        self.enc_dim = embed_dim
        # ====================================================

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        # --- Perceptual(VGG) 준비: 2D feature extractor 고정 ---
        if self.use_perc:
            from torchvision.models import vgg16
            vgg = vgg16(weights='IMAGENET1K_V1').features.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            # 여러 스케일의 feature map을 쓰고 싶으면 필요한 블록까지만 잘라서 보관
            # (예: relu1_2, relu2_2, relu3_3 등 레이어 인덱스는 필요에 맞게 조정)
            self.vgg_feat = nn.Sequential(*list(vgg.children())[:16])  # 예시

        # --- 2D Sobel 커널 준비(엣지) ---
        if self.use_edge:
            sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32)
            sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32)
            self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
            self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))

        # ---- SimSiam-style projector (2-layer MLP) ----
        # BN1d는 배치 크기 >=2 권장. 배치가 작으면 LayerNorm으로 바꾸세요.
        if self.use_contrastive:
            hd = proj_hidden_dim or self.enc_dim
            self.projector = nn.Sequential(
                nn.Linear(self.enc_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(inplace=True),
                nn.Linear(hd, self.enc_dim)
            )


    # 2) 마스크를 이미지 해상도로 올리는 유틸(옵션)

    # Perceptual/Edge를 마스크된 패치에만 집중하고 싶다면, [N,L] → [N,1,H,W]로 업샘플한 mask를 만들면 됩니다.

    def mask_to_img(self, mask):
        # mask: [N, L]  (L = (H/p)*(W/p))
        p = self.patch_embed.patch_size[0]
        H = W = int((mask.shape[1]) ** 0.5) * p
        N = mask.shape[0]
        grid = mask.view(N, int(H/p), int(W/p))  # [N, h, w]
        grid = grid.unsqueeze(1).repeat(1, 1, p, p)  # [N, 1, h, w] -> repeat to pixels
        grid = grid.view(N, 1, H, W)  # [N,1,H,W], 0=keep, 1=masked
        return grid

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # === (변경) 마스킹 제거: 전 패치 유지 ===
        N, L, D = x.shape
        mask = torch.zeros(N, L, device=x.device)  # 모두 keep
        ids_restore = torch.arange(L, device=x.device).unsqueeze(0).repeat(N, 1)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    # ============== 3) Perceptual/Edge 손실 계산 함수 ==================
    def perceptual_loss(self, x, y):
    # x,y: [N,3,H,W], range는 동일 스케일 가정 (norm_pix_loss=False 권장)
        with torch.no_grad():
            self.vgg_feat.eval()
        fx = self.vgg_feat(x)
        fy = self.vgg_feat(y)

        return torch.mean((fx - fy) ** 2)

    def edge_map(self, img):
        # img: [N,3,H,W] -> Gray로 변환 후 Sobel
        gray = 0.299*img[:,0:1] + 0.587*img[:,1:2] + 0.114*img[:,2:3]
        gx = torch.nn.functional.conv2d(gray, self.sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy + 1e-6)

        return mag

    def edge_loss(self, x, y):
        ex = self.edge_map(x)
        ey = self.edge_map(y)

        return torch.mean((ex - ey) ** 2)
    # ================================================================

    # 4) forward_loss()에 합산
    def forward_loss(self, imgs, pred, mask):
        # --- 기존 픽셀 재구성 손실(MSE on masked patches) ---
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            # Perceptual/Edge를 쓰려면 False 권장. True면 스케일이 어긋납니다.
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss_pix = (pred - target) ** 2
        loss_pix = loss_pix.mean(dim=-1)             # [N, L]
        # loss_pix = (loss_pix * mask).sum() / mask.sum()

        # === (변경) 마스크 가중 대신 전 패치 평균 ===
        loss_pix = loss_pix.mean()

        total = loss_pix

        # --- Perceptual / Edge 손실 (이미지 공간에서) ---
        if self.use_perc or self.use_edge:
            # pred -> 이미지 복원
            recon = self.unpatchify(pred)            # [N,3,H,W]
            # 필요하면 mask를 이미지 크기로 올려서 마스킹된 영역만 평균
            mask_img = self.mask_to_img(mask)        # [N,1,H,W]; 0 keep, 1 masked

            if self.use_perc:
                lp = self.perceptual_loss(recon, imgs)
                # 마스킹 영역만 보려면 아래처럼 가중 평균으로 바꿔도 됩니다.
                # lp = ((self.vgg_feat(recon)-self.vgg_feat(imgs))**2).mean(dim=(1,2,3))
                # lp = (lp * mask_img.mean(dim=(2,3)).squeeze(1)).mean()  # 예시
                total = total + self.lambda_perc * lp

            if self.use_edge:
                le = self.edge_loss(recon, imgs)
                total = total + self.lambda_edge * le

        return total

    def forward(self, imgs, mask_ratio=0.0):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = ViT_AEpp(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = ViT_AEpp(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = ViT_AEpp(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

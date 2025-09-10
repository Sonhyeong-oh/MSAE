# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import torch.nn.functional as F
import torch

import util.misc as misc
import util.lr_sched as lr_sched


def _cosine_loss(p, z):
    # SimSiam-style: z는 stop-grad
    z = z.detach()
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return - (p * z).sum(dim=-1).mean()

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # 1) 재구성(픽셀/퍼셉추얼/엣지 포함) 손실: 모델이 반환
            rec_loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
            total_loss = rec_loss
            cl_loss = torch.zeros((), device=device)

            # 2) Contrastive 손실(옵션): 두 개의 서로 다른 마스크 뷰 인코딩
            if getattr(args, 'use_contrastive', False):
                # forward_encoder는 (latent, mask, ids_restore) 반환, latent[:,0]이 CLS
                z1, _, _ = model.forward_encoder(samples, mask_ratio=args.mask_ratio)  # view-1
                z2, _, _ = model.forward_encoder(samples, mask_ratio=args.mask_ratio)  # view-2 (다른 mask; 내부 rand로 differ)

                z1 = z1[:, 0]  # [N, D]
                z2 = z2[:, 0]

                # projector 통과
                p1 = model.projector(z1)
                p2 = model.projector(z2)

                # SimSiam-style 양방향 코사인 손실
                cl_loss = 0.5 * (_cosine_loss(p1, z2) + _cosine_loss(p2, z1))

                # 최종 손실 합산 (λ_cl 가중치)
                total_loss = total_loss + args.lambda_cl * cl_loss

        loss_value = total_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        total_loss = total_loss / accum_iter
        loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        if getattr(args, 'use_contrastive', False):
            metric_logger.update(loss_rec=rec_loss.item())
            metric_logger.update(loss_cl=cl_loss.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss_total', loss_value_reduce, epoch_1000x)
            if getattr(args, 'use_contrastive', False):
                log_writer.add_scalar('train_loss_rec', misc.all_reduce_mean(rec_loss.item()), epoch_1000x)
                log_writer.add_scalar('train_loss_cl', misc.all_reduce_mean(cl_loss.item()), epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
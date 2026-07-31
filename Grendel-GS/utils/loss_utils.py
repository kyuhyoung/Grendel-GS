#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# ── fused SSIM (Taming 3DGS) ─────────────────────────────────────────────
# 동일 수식(11x11 gaussian window, same padding, C1=0.01^2, C2=0.03^2)의 융합 커널.
# 목적: SSIM 임시 버퍼로 인한 거짓 Cat 1 OOM 제거 + iteration 속도 향상.
# FUSED_SSIM=1 이고 import 가 성공했을 때만 사용, 실패 시 기존 경로로 자동 폴백.
import os as _os

_fused_ssim_fn = None
if _os.environ.get("FUSED_SSIM", "0") == "1":
    try:
        from fused_ssim import fused_ssim as _fused_ssim_fn
        print("[fused-ssim] enabled (FUSED_SSIM=1, import OK)", flush=True)
    except Exception as _e:  # 미설치/컴파일 불일치 등 — 무인 운영 우선, 조용히 폴백
        print(f"[fused-ssim] unavailable ({_e}) - falling back to legacy SSIM", flush=True)
        _fused_ssim_fn = None


def fused_ssim_sum(img1, img2):
    """all-ones 마스크의 pixelwise_ssim_with_mask(...).sum() 과 동치인 fused 계산.

    fused_ssim 은 same-padding SSIM map 의 mean 을 반환하므로 numel 을 곱해
    sum 스케일로 되돌린다. 사용 불가 조건이면 None 을 반환 (호출부에서 폴백).
    """
    if _fused_ssim_fn is None:
        return None
    # fused-ssim 제약: 4D (B,C,H,W) float CUDA 텐서, 윈도우보다 작은 이미지는 제외
    if img1.dim() != 3 or not img1.is_cuda or img1.shape[-1] < 11 or img1.shape[-2] < 11:
        return None
    try:
        mean_ssim = _fused_ssim_fn(
            img1.unsqueeze(0).contiguous(), img2.unsqueeze(0).contiguous()
        )
        return mean_ssim * img1.numel()
    except torch.cuda.OutOfMemoryError:
        # 메모리 부족은 폴백하지 않고 그대로 올려보낸다 (2026-07-24 실측).
        # legacy 경로는 fused 보다 메모리를 더 쓰므로, fused 가 OOM 났으면 legacy 는
        # 반드시 또 OOM 난다 — 폴백해봐야 헛수고이고 더 큰 할당을 시도해 상황을 악화시킨다.
        # 그대로 올려보내면 기존 OOM 처리기가 카테고리를 판정해 재시도/분할한다.
        raise
    except Exception as e:
        # 미설치·형상 불일치 등 진짜 "쓸 수 없는" 경우만 폴백
        print(f"[fused-ssim] runtime error ({e}) - falling back to legacy SSIM", flush=True)
        return None


def pixelwise_l1_with_mask(img1, img2, pixel_mask):
    # img1, img2: (3, H, W)
    # pixel_mask: (H, W) bool torch tensor as mask.
    # only compute l1 loss for the pixels that are touched

    pixelwise_l1_loss = torch.abs((img1 - img2)) * pixel_mask.unsqueeze(0)
    return pixelwise_l1_loss


def pixelwise_ssim_with_mask(img1, img2, pixel_mask):
    window_size = 11

    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    pixelwise_ssim_loss = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    pixelwise_ssim_loss = pixelwise_ssim_loss * pixel_mask.unsqueeze(0)

    return pixelwise_ssim_loss

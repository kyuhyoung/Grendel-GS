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

from torch.utils.checkpoint import checkpoint


def pixelwise_ssim_with_mask_improved_chunked(img1, img2, pixel_mask, chunk_size):
    """경계 처리를 개선한 청크 버전"""
    
    C, H, W = img1.shape
    window_size = 11
    pad = window_size // 2
    
    # 결과 텐서 초기화
    result = torch.zeros((1, H, W), dtype=img1.dtype, device=img1.device)
    overlap_count = torch.zeros((H, W), dtype=torch.float32, device=img1.device)
    
    # 겹치는 청크로 처리 (경계 문제 완화)
    stride = chunk_size - 2 * pad  # 겹치는 영역 고려
    
    for i in range(0, H, stride):
        for j in range(0, W, stride):
            # 청크 범위 계산
            h_start = i
            h_end = min(i + chunk_size, H)
            w_start = j  
            w_end = min(j + chunk_size, W)
            
            # 실제 청크 크기가 너무 작으면 건너뛰기
            if (h_end - h_start) < window_size or (w_end - w_start) < window_size:
                continue
            
            # 청크 추출
            chunk1 = img1[:, h_start:h_end, w_start:w_end]
            chunk2 = img2[:, h_start:h_end, w_start:w_end] 
            chunk_mask = pixel_mask[h_start:h_end, w_start:w_end]
            
            # 청크에 대해 SSIM 계산 (FP32로)
            #chunk_result = pixelwise_ssim_single_chunk_fp32(chunk1, chunk2, chunk_mask)
            chunk_result = pixelwise_ssim_with_mask_original(chunk1, chunk2, chunk_mask)
            
            # 경계 영역 제외하고 결과 누적
            inner_h_start = pad if i > 0 else 0
            inner_w_start = pad if j > 0 else 0
            inner_h_end = chunk_result.shape[1] - pad if (i + chunk_size) < H else chunk_result.shape[1]
            inner_w_end = chunk_result.shape[2] - pad if (j + chunk_size) < W else chunk_result.shape[2]
            
            result_h_start = h_start + inner_h_start
            result_h_end = h_start + inner_h_end
            result_w_start = w_start + inner_w_start  
            result_w_end = w_start + inner_w_end
            
            result[:, result_h_start:result_h_end, result_w_start:result_w_end] += \
                chunk_result[:, inner_h_start:inner_h_end, inner_w_start:inner_w_end]
            
            overlap_count[result_h_start:result_h_end, result_w_start:result_w_end] += 1
            
            # 메모리 정리
            del chunk1, chunk2, chunk_mask, chunk_result
            torch.cuda.empty_cache()
    
    # 겹치는 영역 평균화
    overlap_count = torch.clamp(overlap_count, min=1)
    result = result / overlap_count.unsqueeze(0)
    
    return result
'''
def pixelwise_ssim_single_chunk_fp32(img1, img2, pixel_mask):
    """FP32로 정확한 청크 SSIM 계산"""
    # 배치 차원 추가
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    
    # 원본 함수와 동일한 FP32 계산
    window_size = 11
    channel = img1.size(1)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    # ... (기존 SSIM 계산 로직, FP32로)
    # 수치적 안정성을 위한 클램핑 포함
    
    return result
'''

def pixelwise_ssim_with_mask_safe_efficient(img1, img2, pixel_mask):
    """메모리 효율적이면서 수치적으로 안전한 버전"""
    
    window_size = 11
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    # 계산을 단계별로 나누어 메모리 절약
    with torch.no_grad():
        # 1단계: 평균 계산
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    # gradient가 필요한 부분만 별도 계산
    mu1.requires_grad_(img1.requires_grad)
    mu2.requires_grad_(img2.requires_grad)
    
    # 나머지 계산들...
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    # 메모리 해제
    del mu1, mu2
    torch.cuda.empty_cache()
    
    # 분산 계산
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    # 안전한 상수 사용 (더 큰 값으로)
    C1 = (0.01 * 2)**2  # 좀 더 큰 값
    C2 = (0.03 * 2)**2
    
    # 수치적 안정성을 위한 클램핑
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    denominator = torch.clamp(denominator, min=1e-8)  # 0으로 나누기 방지
    
    pixelwise_ssim_loss = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / denominator
    pixelwise_ssim_loss = pixelwise_ssim_loss * pixel_mask.unsqueeze(0)
    
    return pixelwise_ssim_loss

def pixelwise_ssim_with_mask_checkpointed(img1, img2, pixel_mask):
    """Gradient checkpointing으로 메모리 절약하면서 FP32 유지"""
    
    def ssim_forward(img1, img2, pixel_mask):
        # 원본 함수와 동일하지만 중간 결과를 저장하지 않음
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
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        pixelwise_ssim_loss = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        pixelwise_ssim_loss = pixelwise_ssim_loss * pixel_mask.unsqueeze(0)
        return pixelwise_ssim_loss
    
    # Gradient checkpointing 사용
    return checkpoint(ssim_forward, img1, img2, pixel_mask, use_reentrant=False)


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


def pixelwise_l1_with_mask(img1, img2, pixel_mask):
    # img1, img2: (3, H, W)
    # pixel_mask: (H, W) bool torch tensor as mask.
    # only compute l1 loss for the pixels that are touched
    #print(f'img1.shape : {img1.shape}, img2.shape : {img2.shape}'); #exit(1)
    pixelwise_l1_loss = torch.abs((img1 - img2)) * pixel_mask.unsqueeze(0)
    return pixelwise_l1_loss



def pixelwise_ssim_with_mask_fp16(img1, img2, pixel_mask):
    """반정밀도를 사용하여 메모리 사용량 절반으로 감소"""
    
    # 원본 dtype 저장
    original_dtype = img1.dtype
    
    # 반정밀도로 변환
    img1_fp16 = img1.half()
    img2_fp16 = img2.half()
    
    window_size = 11
    channel = img1_fp16.size(-3)
    window = create_window(window_size, channel).half()
    if img1_fp16.is_cuda:
        window = window.cuda(img1_fp16.get_device())
    
    # SSIM 계산 (반정밀도로)
    mu1 = F.conv2d(img1_fp16, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2_fp16, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1_fp16 * img1_fp16, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2_fp16 * img2_fp16, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1_fp16 * img2_fp16, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    pixelwise_ssim_loss = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    
    # 마스크 적용
    pixelwise_ssim_loss = pixelwise_ssim_loss * pixel_mask.unsqueeze(0).half()
    
    # 원본 dtype으로 복원
    return pixelwise_ssim_loss.to(original_dtype)


def pixelwise_ssim_with_mask_mem_eff(img1, img2, pixel_mask):
    window_size = 11
    channel = img1.size(-3)
    
    # 윈도우를 미리 생성하고 재사용
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    # 메모리 절약을 위해 중간 계산들을 즉시 삭제
    with torch.cuda.amp.autocast(enabled=False):  # 혼합 정밀도 비활성화
        # 평균 계산
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        
        # 제곱 계산 (inplace 연산 사용)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        # 메모리 해제
        del mu1, mu2
        torch.cuda.empty_cache()
        
        # 분산 계산
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
        sigma1_sq -= mu1_sq
        
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
        sigma2_sq -= mu2_sq
        
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        sigma12 -= mu1_mu2
        
        # 상수 정의
        C1 = 0.01**2
        C2 = 0.03**2
        
        # SSIM 계산 (메모리 효율적으로)
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        # 중간 텐서들 삭제
        del mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12
        torch.cuda.empty_cache()
        
        pixelwise_ssim_loss = numerator / denominator
        del numerator, denominator
        
        # 마스크 적용
        if pixel_mask.dim() == 2:  # 2D 마스크인 경우
            pixel_mask = pixel_mask.unsqueeze(0).unsqueeze(0)
        elif pixel_mask.dim() == 3:  # 3D 마스크인 경우
            pixel_mask = pixel_mask.unsqueeze(0)
            
        pixelwise_ssim_loss = pixelwise_ssim_loss * pixel_mask
        
        return pixelwise_ssim_loss

def pixelwise_ssim_with_mask_ori(img1, img2, pixel_mask):
    
    #print(f'img1.shape : {img1.shape}, img2.shape : {img2.shape}'); #exit(1)
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


def pixelwise_ssim_single_chunk(img1, img2, pixel_mask):
    """단일 청크에 대한 SSIM 계산 - (C, H, W) 입력 형태"""
    
    window_size = 11
    channel = img1.size(0)  # 첫 번째 차원이 채널
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    # (C, H, W) -> (1, C, H, W) 배치 차원 추가
    img1_batch = img1.unsqueeze(0)
    img2_batch = img2.unsqueeze(0)
    
    # 메모리 효율적인 SSIM 계산
    with torch.cuda.amp.autocast(enabled=False):
        # 평균 계산
        mu1 = F.conv2d(img1_batch, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2_batch, window, padding=window_size // 2, groups=channel)
        
        # 제곱 계산
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        # 메모리 해제
        del mu1, mu2
        torch.cuda.empty_cache()
        
        # 분산 계산
        sigma1_sq = F.conv2d(img1_batch * img1_batch, window, padding=window_size // 2, groups=channel)
        sigma1_sq -= mu1_sq
        
        sigma2_sq = F.conv2d(img2_batch * img2_batch, window, padding=window_size // 2, groups=channel)
        sigma2_sq -= mu2_sq
        
        sigma12 = F.conv2d(img1_batch * img2_batch, window, padding=window_size // 2, groups=channel)
        sigma12 -= mu1_mu2
        
        # 상수 정의
        C1 = 0.01**2
        C2 = 0.03**2
        
        # SSIM 계산
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        # 중간 텐서들 삭제
        del mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12
        torch.cuda.empty_cache()
        
        pixelwise_ssim_loss = numerator / denominator
        del numerator, denominator
        
        # 마스크 적용 - pixel_mask는 (H, W) 형태
        # (1, C, H, W) * (1, 1, H, W) = (1, C, H, W)
        pixel_mask_expanded = pixel_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        pixelwise_ssim_loss = pixelwise_ssim_loss * pixel_mask_expanded
        
        # 첫 번째 채널만 반환 (1, H, W)
        return pixelwise_ssim_loss[:, :1, :, :]


def pixelwise_ssim_with_mask_simple_chunked(img1, img2, pixel_mask, chunk_size=256):
    """간단한 청크 단위 SSIM 계산"""
    
    C, H, W = img1.shape
    result = torch.zeros((1, H, W), dtype=img1.dtype, device=img1.device)
    
    # 겹치지 않는 청크로 처리
    for i in range(0, H, chunk_size):
        for j in range(0, W, chunk_size):
            h_end = min(i + chunk_size, H)
            w_end = min(j + chunk_size, W)
            
            # 청크 추출
            chunk1 = img1[:, i:h_end, j:w_end]
            chunk2 = img2[:, i:h_end, j:w_end]
            chunk_mask = pixel_mask[i:h_end, j:w_end]
            
            # 원본 함수 호출 (작은 청크에 대해)
            #chunk_result = pixelwise_ssim_with_mask_original(chunk1, chunk2, chunk_mask)
            chunk_result = pixelwise_ssim_single_chunk(chunk1, chunk2, chunk_mask)
            
            result[:, i:h_end, j:w_end] = chunk_result
            # 메모리 정리
            del chunk1, chunk2, chunk_mask, chunk_result
            torch.cuda.empty_cache()
    
    return result

def pixelwise_ssim_with_mask_original(img1, img2, pixel_mask):
    """원본 함수를 (C, H, W) 입력에 맞게 수정"""
    
    # 배치 차원 추가
    img1 = img1.unsqueeze(0)  # (1, C, H, W)
    img2 = img2.unsqueeze(0)  # (1, C, H, W)
    
    window_size = 11
    channel = img1.size(1)  # 두 번째 차원이 채널 (배치 추가 후)
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
    
    # 마스크 적용 - (1, C, H, W) * (1, 1, H, W)
    pixelwise_ssim_loss = pixelwise_ssim_loss * pixel_mask.unsqueeze(0).unsqueeze(0)
    
    # 첫 번째 채널만 반환하고 배치 차원 제거
    return pixelwise_ssim_loss[:, :1, :, :].squeeze(0)  # (1, H, W)

def pixelwise_ssim_with_mask_mixed_precision(img1, img2, pixel_mask):
    """혼합 정밀도를 사용하되 중요한 계산은 FP32로"""
    
    # 입력만 FP16으로 변환하여 메모리 절약
    img1_fp16 = img1.half()
    img2_fp16 = img2.half()
    
    window_size = 11
    channel = img1_fp16.size(-3)
    window = create_window(window_size, channel).half()
    if img1_fp16.is_cuda:
        window = window.cuda(img1_fp16.get_device())
    
    # Convolution은 FP16으로 (메모리 절약)
    mu1 = F.conv2d(img1_fp16, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2_fp16, window, padding=window_size // 2, groups=channel)
    
    # 중요한 계산들은 FP32로 변환
    mu1 = mu1.float()
    mu2 = mu2.float()
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 분산 계산도 FP32로
    sigma1_sq = F.conv2d(img1_fp16 * img1_fp16, window, padding=window_size // 2, groups=channel).float() - mu1_sq
    sigma2_sq = F.conv2d(img2_fp16 * img2_fp16, window, padding=window_size // 2, groups=channel).float() - mu2_sq
    sigma12 = F.conv2d(img1_fp16 * img2_fp16, window, padding=window_size // 2, groups=channel).float() - mu1_mu2
    
    # SSIM 계산은 FP32로 (수치적 안정성)
    C1 = 0.01**2
    C2 = 0.03**2
    
    pixelwise_ssim_loss = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    
    # 마스크 적용
    pixelwise_ssim_loss = pixelwise_ssim_loss * pixel_mask.unsqueeze(0).float()
    
    return pixelwise_ssim_loss

def pixelwise_ssim_with_mask(img1, img2, pixel_mask):
    #return pixelwise_ssim_with_mask_ori(img1, img2, pixel_mask)
    #return pixelwise_ssim_with_mask_mem_eff(img1, img2, pixel_mask)
    #return pixelwise_ssim_with_mask_checkpointed(img1, img2, pixel_mask)
    #return pixelwise_ssim_with_mask_safe_efficient(img1, img2, pixel_mask)
    #return pixelwise_ssim_with_mask_mixed_precision(img1, img2, pixel_mask)
    return pixelwise_ssim_with_mask_improved_chunked(img1, img2, pixel_mask, 4096)
    #return pixelwise_ssim_with_mask_improved_chunked(img1, img2, pixel_mask, 256)
    #return pixelwise_ssim_with_mask_fp16(img1, img2, pixel_mask)
    #return pixelwise_ssim_with_mask_simple_chunked(img1, img2, pixel_mask, 256)


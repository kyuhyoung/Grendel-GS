#!/usr/bin/env python3

import torch
import torch.nn.functional as F

def compute_ssim_numerator_efficient(mu1_mu2, sigma12, C1=0.01**2, C2=0.03**2):
    """
    메모리 효율적인 numerator 계산
    In-place 연산을 최대한 활용
    """
    # (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)를 메모리 효율적으로 계산

    # Step 1: mu1_mu2를 2배로 (in-place)
    mu1_mu2.mul_(2)
    # Step 2: C1 더하기 (in-place)
    mu1_mu2.add_(C1)

    # Step 3: sigma12를 2배로 (in-place)
    sigma12.mul_(2)
    # Step 4: C2 더하기 (in-place)
    sigma12.add_(C2)

    # Step 5: 두 텐서 곱하기 (mu1_mu2를 결과로 재사용)
    mu1_mu2.mul_(sigma12)

    return mu1_mu2  # 이제 이것이 numerator


def pixelwise_ssim_memory_efficient(img1, img2, pixel_mask, window_size=11):
    """
    메모리 최적화된 SSIM 계산
    - In-place 연산 최대 활용
    - 중간 텐서 재사용
    - 즉시 메모리 해제
    """
    channel = img1.size(-3)

    # Window 생성
    sigma = 1.5
    gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
                          for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device, dtype=img1.dtype)

    padding = window_size // 2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 평균 계산
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # mu1, mu2는 더 이상 필요 없음
    del mu1, mu2

    # 분산 계산 (in-place로)
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel)
    sigma1_sq.sub_(mu1_sq)  # in-place subtraction

    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel)
    sigma2_sq.sub_(mu2_sq)  # in-place subtraction

    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel)
    sigma12.sub_(mu1_mu2)  # in-place subtraction

    # Numerator 계산 (in-place, mu1_mu2와 sigma12를 재사용)
    # numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    mu1_mu2_copy = mu1_mu2.clone()  # denominator를 위해 복사본 하나 필요
    sigma12_copy = sigma12.clone()

    # In-place로 numerator 계산
    mu1_mu2.mul_(2).add_(C1)  # 2 * mu1_mu2 + C1
    sigma12.mul_(2).add_(C2)  # 2 * sigma12 + C2
    numerator = mu1_mu2.mul_(sigma12)  # 최종 numerator

    # Denominator 계산 (in-place)
    # denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    mu1_sq.add_(mu2_sq).add_(C1)  # mu1_sq + mu2_sq + C1
    sigma1_sq.add_(sigma2_sq).add_(C2)  # sigma1_sq + sigma2_sq + C2
    denominator = mu1_sq.mul_(sigma1_sq)  # 최종 denominator

    # 최종 SSIM 계산
    ssim_map = numerator.div_(denominator)  # in-place division

    # 마스크 적용 (in-place)
    if pixel_mask is not None:
        ssim_map.mul_(pixel_mask.view(1, *pixel_mask.shape))

    return ssim_map


def pixelwise_ssim_chunked(img1, img2, pixel_mask, chunk_size=1024):
    """
    이미지를 청크로 나눠서 처리하여 메모리 사용량 감소
    """
    H, W = img1.shape[-2:]
    result = torch.zeros_like(img1)

    for i in range(0, H, chunk_size):
        for j in range(0, W, chunk_size):
            i_end = min(i + chunk_size, H)
            j_end = min(j + chunk_size, W)

            # 청크 추출
            img1_chunk = img1[..., i:i_end, j:j_end]
            img2_chunk = img2[..., i:i_end, j:j_end]
            mask_chunk = pixel_mask[i:i_end, j:j_end] if pixel_mask is not None else None

            # 청크별로 SSIM 계산
            ssim_chunk = pixelwise_ssim_memory_efficient(img1_chunk, img2_chunk, mask_chunk)

            # 결과에 저장
            result[..., i:i_end, j:j_end] = ssim_chunk

            # 메모리 정리
            del ssim_chunk
            torch.cuda.empty_cache()

    return result
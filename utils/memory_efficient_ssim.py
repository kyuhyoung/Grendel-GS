#!/usr/bin/env python3

import torch
import torch.nn.functional as F

def pixelwise_ssim_ultra_efficient(img1, img2, pixel_mask, window_size=11):
    """
    Ultra memory-efficient SSIM calculation
    - Uses minimal intermediate tensors
    - Aggressive in-place operations
    - Immediate tensor deletion
    """
    channel = img1.size(-3)

    # Create window once and reuse
    sigma = 1.5
    gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
                          for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device, dtype=img1.dtype)

    # Constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    padding = window_size // 2

    # Calculate means with immediate reuse
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    # Calculate products in-place where possible
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Clear original means
    del mu1, mu2
    torch.cuda.empty_cache()

    # Calculate variances
    sigma1_sq = F.conv2d(img1.pow(2), window, padding=padding, groups=channel)
    sigma1_sq -= mu1_sq  # In-place subtraction

    sigma2_sq = F.conv2d(img2.pow(2), window, padding=padding, groups=channel)
    sigma2_sq -= mu2_sq  # In-place subtraction

    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel)
    sigma12 -= mu1_mu2  # In-place subtraction

    # Calculate SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    # Clear intermediate tensors
    del mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12
    torch.cuda.empty_cache()

    ssim_map = numerator / denominator
    del numerator, denominator

    # Apply mask in-place if provided
    if pixel_mask is not None:
        ssim_map *= pixel_mask.view(1, *pixel_mask.shape)

    return ssim_map

def apply_mask_memory_efficient(tensor, mask):
    """
    Apply mask to tensor with minimal memory usage
    """
    if mask is None:
        return tensor

    # Get dimensions
    t_shape = tensor.shape
    m_shape = mask.shape

    # Calculate broadcast dimensions
    broadcast_dims = [1] * (len(t_shape) - len(m_shape)) + list(m_shape)

    # Use in-place operation with reshaped mask
    tensor *= mask.reshape(broadcast_dims)
    return tensor
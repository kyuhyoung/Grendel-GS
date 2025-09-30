#!/usr/bin/env python3

"""
SSIM 연산의 메모리 사용량 상세 분석
"""

import numpy as np

def analyze_memory(height=4096, width=4096, channels=3, dtype_bytes=4):
    """
    pixelwise_ssim_loss = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / denominator
    연산의 메모리 사용량 분석
    """

    pixels = height * width

    print(f"이미지 크기: {height}x{width}, 채널: {channels}")
    print(f"데이터 타입: float32 ({dtype_bytes} bytes)")
    print("=" * 60)

    # 입력 텐서들
    tensor_size = channels * pixels * dtype_bytes / (1024**2)
    print(f"\n입력 텐서 (각각):")
    print(f"  mu1_mu2: ({channels}, {height}, {width}) = {tensor_size:.1f} MiB")
    print(f"  sigma12: ({channels}, {height}, {width}) = {tensor_size:.1f} MiB")
    print(f"  mu1_sq:  ({channels}, {height}, {width}) = {tensor_size:.1f} MiB")
    print(f"  mu2_sq:  ({channels}, {height}, {width}) = {tensor_size:.1f} MiB")
    print(f"  sigma1_sq: ({channels}, {height}, {width}) = {tensor_size:.1f} MiB")
    print(f"  sigma2_sq: ({channels}, {height}, {width}) = {tensor_size:.1f} MiB")

    print(f"\n중간 계산 과정:")
    print("-" * 40)

    # Step 1: 2 * mu1_mu2
    print(f"1. temp1 = 2 * mu1_mu2")
    print(f"   새 텐서 생성: {tensor_size:.1f} MiB")
    total = tensor_size

    # Step 2: temp1 + C1
    print(f"2. temp2 = temp1 + C1")
    print(f"   새 텐서 생성: {tensor_size:.1f} MiB")
    total += tensor_size

    # Step 3: 2 * sigma12
    print(f"3. temp3 = 2 * sigma12")
    print(f"   새 텐서 생성: {tensor_size:.1f} MiB")
    total += tensor_size

    # Step 4: temp3 + C2
    print(f"4. temp4 = temp3 + C2")
    print(f"   새 텐서 생성: {tensor_size:.1f} MiB")
    total += tensor_size

    # Step 5: temp2 * temp4 (numerator)
    print(f"5. numerator = temp2 * temp4")
    print(f"   새 텐서 생성: {tensor_size:.1f} MiB")
    total += tensor_size

    print(f"\n이 시점까지 추가 메모리: {total:.1f} MiB")

    # Denominator 계산도 비슷하게 메모리 사용
    print(f"\ndenominator 계산도 유사한 메모리 필요")
    print(f"최종 나눗셈: numerator / denominator = {tensor_size:.1f} MiB")

    print(f"\n총 추가 메모리 요구량: ~{total * 1.5:.0f} MiB")

    print("\n=" * 60)
    print("문제점:")
    print("1. 각 연산이 새로운 텐서를 생성")
    print("2. PyTorch는 autograd를 위해 중간 결과 저장")
    print("3. GPU 메모리가 거의 가득 찬 상태에서는 작은 할당도 실패")

    print("\n해결 방법:")
    print("1. torch.no_grad() 사용 (gradient 불필요한 경우)")
    print("2. In-place 연산 사용")
    print("3. 중간 변수 재사용")
    print("4. Mixed precision (FP16) 사용")

if __name__ == "__main__":
    # 4K 이미지 기준
    analyze_memory(4096, 4096, 3)

    print("\n" + "=" * 60)
    print("8K 이미지의 경우:")
    print("-" * 40)
    analyze_memory(8192, 8192, 3)
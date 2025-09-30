#!/usr/bin/env python3

"""
CUDA OOM 문제 해결 방법 설명
"""

print("=" * 60)
print("CUDA Out of Memory 해결")
print("=" * 60)
print()
print("문제:")
print("pixelwise_ssim_loss * pixel_mask.unsqueeze(0) 연산에서 OOM 발생")
print("GPU에 23.43 GiB 사용 중, 23 MiB만 남은 상태")
print()
print("원인:")
print("1. unsqueeze(0)가 새로운 텐서 뷰를 생성")
print("2. 곱셈 연산이 추가 메모리 할당")
print("3. 메모리 단편화로 연속된 748 MiB 할당 불가능")
print()
print("해결책:")
print("1. view() 사용으로 메모리 복사 없이 reshape")
print("   변경 전: pixel_mask.unsqueeze(0)")
print("   변경 후: pixel_mask.view(1, *pixel_mask.shape)")
print()
print("2. In-place 연산 고려")
print("   pixelwise_ssim_loss.mul_(mask) # 새 텐서 생성 없음")
print()
print("3. 추가 최적화:")
print("   - torch.cuda.empty_cache() 호출")
print("   - gradient checkpointing 사용")
print("   - mixed precision (FP16) 사용")
print()
print("=" * 60)
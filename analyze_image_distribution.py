#!/usr/bin/env python3

"""
GPU 병렬 처리에서 이미지 분할 분석
"""

print("=" * 60)
print("GPU 병렬 처리에서 이미지 분할 방식")
print("=" * 60)
print()

print("CUDA 코드 분석 (auxiliary.h):")
print("-" * 40)
print()
print("1. getLocalTileRect 함수:")
print("   - grid.x를 world_size로 나누어 각 GPU별 타일 범위 결정")
print("   - chunk_size = grid.x / world_size")
print("   - 나머지가 있으면 일부 GPU가 추가 타일 처리")
print()
print("2. getLocalPixelRect 함수:")
print("   - 타일 범위를 픽셀 범위로 변환")
print("   - local_pixel_rect_min.x = local_tile_rect_min.x * BLOCK_X")
print("   - local_pixel_rect_max.x = min(image_width, local_tile_rect_max.x * BLOCK_X)")
print()

def calculate_distribution(W, H, G, BLOCK_X=16, BLOCK_Y=16):
    """
    이미지 분할 계산
    W: 이미지 width
    H: 이미지 height
    G: GPU 개수
    """
    print(f"예시: W={W}, H={H}, GPU 개수={G}")
    print("-" * 40)

    # 타일 개수 계산
    TILE_X = (W + BLOCK_X - 1) // BLOCK_X
    TILE_Y = (H + BLOCK_Y - 1) // BLOCK_Y

    print(f"전체 타일 개수: TILE_X={TILE_X}, TILE_Y={TILE_Y}")

    # 각 GPU별 타일 분할 (X 방향으로)
    chunk_size = TILE_X // G
    chunk_remain = TILE_X % G

    print(f"기본 chunk_size: {chunk_size}")
    print(f"나머지 타일: {chunk_remain}")
    print()

    for rank in range(G):
        if rank < chunk_remain:
            xl = chunk_size * rank + rank
            xr = chunk_size * (rank + 1) + rank + 1
        else:
            xl = chunk_size * rank + chunk_remain
            xr = chunk_size * (rank + 1) + chunk_remain

        # 픽셀 범위로 변환
        pixel_xl = xl * BLOCK_X
        pixel_xr = min(W, xr * BLOCK_X)

        actual_width = pixel_xr - pixel_xl
        actual_height = H  # Y 방향은 분할하지 않음

        print(f"GPU {rank}: 타일 [{xl}, {xr}), 픽셀 [{pixel_xl}, {pixel_xr})")
        print(f"         실제 크기: {actual_width} x {actual_height}")
        print(f"         메모리 (float32): {actual_width * actual_height * 3 * 4 / (1024**2):.1f} MiB")
        print()

print("분할 방식:")
print("-" * 40)
print("✓ X 방향(width)으로 타일 분할")
print("✓ Y 방향(height)은 분할하지 않음")
print("✓ 각 GPU가 처리하는 이미지: 실제로는 (W/G) x H 정도")
print()

calculate_distribution(4096, 4096, 4)
print("=" * 60)
calculate_distribution(8192, 4096, 8)
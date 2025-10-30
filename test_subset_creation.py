#!/usr/bin/env python3
"""
DNQ Subset Creation 테스트 스크립트
"""

import sys
from pathlib import Path
from subset_creator import create_subsets_with_footprints

def test_subset_creation():
    """Samsung_SN_30 데이터셋으로 subset 생성 테스트"""
    
    # 기본 경로 설정
    source_path = Path("/data/Samsung_SN_30")
    output_path = Path("./test_output_subset")
    
    # 테스트 파라미터
    pixel_threshold_a = 1000000  # 1M 픽셀
    min_max_ratio_d = 0.7
    max_subsets = 4
    
    print("=== DNQ Subset Creation 테스트 ===")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Pixel threshold A: {pixel_threshold_a:,}")
    print(f"Min/Max ratio D: {min_max_ratio_d}")
    print(f"Max subsets: {max_subsets}")
    print()
    
    # 출력 디렉토리 생성
    output_path.mkdir(exist_ok=True)
    
    try:
        # Subset 생성 실행
        subsets = create_subsets_with_footprints(
            source_path=source_path,
            output_path=output_path,
            pixel_threshold_a=pixel_threshold_a,
            min_max_ratio_d=min_max_ratio_d,
            max_subsets=max_subsets
        )
        
        print(f"✓ Subset 생성 성공!")
        print(f"  총 {len(subsets)}개 subset 생성됨")
        
        for i, subset in enumerate(subsets):
            print(f"  Subset {i+1}: {len(subset)}개 이미지")
            
        return True
        
    except Exception as e:
        print(f"✗ Subset 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_subset_creation()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3

"""
새로운 --dir_images와 --dir_sparse 옵션 사용 예제

두 가지 사용 방법:

1. 기존 방식 (source_path 사용):
   - source_path/images/: 이미지 파일들
   - source_path/sparse/0/: COLMAP 파일들

   python train.py --source_path /data/scene

2. 새로운 방식 (직접 디렉토리 지정):
   - dir_images: 이미지 파일들이 있는 디렉토리
   - dir_sparse: images.txt, cameras.txt, points3D.txt가 있는 디렉토리

   python train.py --dir_images /path/to/images --dir_sparse /path/to/sparse

이렇게 하면 images와 sparse 폴더가 다른 위치에 있을 때 발생하는 FileNotFoundError를 해결할 수 있습니다.
"""

print("=" * 60)
print("새로운 디렉토리 옵션 사용법")
print("=" * 60)
print()
print("방법 1: 기존 방식 (하위호환성 유지)")
print("-" * 40)
print("python train.py --source_path /data/scene")
print()
print("디렉토리 구조:")
print("/data/scene/")
print("├── images/")
print("│   ├── 000001.jpg")
print("│   └── 000002.jpg")
print("└── sparse/")
print("    └── 0/")
print("        ├── images.txt")
print("        ├── cameras.txt")
print("        └── points3D.txt")
print()
print("=" * 60)
print()
print("방법 2: 새로운 방식 (직접 디렉토리 지정)")
print("-" * 40)
print("python train.py --dir_images /data/images --dir_sparse /data/temp_initial/sparse/0")
print()
print("디렉토리 구조:")
print("/data/images/")
print("├── 000001.tif")
print("└── 000002.tif")
print()
print("/data/temp_initial/sparse/0/")
print("├── images.txt")
print("├── cameras.txt")
print("└── points3D.txt")
print()
print("=" * 60)
print()
print("당신의 경우 (temp_initial에만 sparse 폴더가 있음):")
print("-" * 40)
print("python train.py \\")
print("    --dir_images /path/to/your/images \\")
print("    --dir_sparse /path/to/temp_initial/sparse/0")
print()
print("이렇게 하면 images와 sparse 폴더가 같은 부모 디렉토리를 공유하지 않아도 됩니다!")
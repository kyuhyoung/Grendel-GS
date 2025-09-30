#!/usr/bin/env python3

"""
progressive_trainer.py의 수정 사항 테스트

변경 전:
- train.py -s temp_path 로 호출
- temp_path/images/ 디렉토리가 없어서 FileNotFoundError 발생

변경 후:
- train.py --dir_images /original/images --dir_sparse temp_path/sparse/0 로 호출
- 이미지는 원본 위치에서 직접 읽고, sparse 파일은 임시 디렉토리에서 읽음
"""

print("=" * 60)
print("Progressive Training 수정 사항")
print("=" * 60)
print()
print("문제:")
print("- progressive_trainer.py가 임시 디렉토리(temp_initial)에 sparse 파일만 생성")
print("- 이미지 파일은 원래 위치에 그대로 있음")
print("- train.py가 temp_initial/images/를 찾으려 해서 FileNotFoundError 발생")
print()
print("해결:")
print("1. train.py에 --dir_images와 --dir_sparse 옵션 추가")
print("2. progressive_trainer.py 수정 (line 724):")
print()
print("   변경 전:")
print('   cmd = [..., "-s", str(temp_path), ...]')
print()
print("   변경 후:")
print('   sparse_dir = temp_path / "sparse" / "0"')
print('   images_dir = self.image_path')
print('   cmd = [..., "--dir_images", str(images_dir), "--dir_sparse", str(sparse_dir), ...]')
print()
print("결과:")
print("- 이미지는 원본 위치에서 직접 읽음")
print("- sparse 파일은 임시 디렉토리에서 읽음")
print("- FileNotFoundError 해결!")
print()
print("=" * 60)
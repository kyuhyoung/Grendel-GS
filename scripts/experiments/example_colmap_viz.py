#!/usr/bin/env python3
"""
COLMAP 시각화 예제 실행 스크립트
실제 COLMAP 데이터로 테스트하기 위한 예제
"""

print("DEBUG: Starting example_colmap_viz.py")
print("DEBUG: Importing modules...")

import os
print("DEBUG: os module imported")

import sys
print("DEBUG: sys module imported")

print("DEBUG: About to import COLMAPVisualizer...")
from colmap_visualizer import COLMAPVisualizer
print("DEBUG: COLMAPVisualizer imported successfully")

def find_colmap_data():
    """현재 Grendel-GS 프로젝트에서 COLMAP 데이터 찾기"""
    print("DEBUG: find_colmap_data() started")
    
    # 환경변수에서 경로 확인
    env_path = os.environ.get('COLMAP_PATH')
    print(f"DEBUG: Environment variable COLMAP_PATH = {env_path}")
    
    if env_path:
        cameras_file = os.path.join(env_path, "cameras.txt")
        print(f"DEBUG: Checking env path cameras.txt: {cameras_file}")
        if os.path.exists(cameras_file):
            print("DEBUG: Found COLMAP data in env path")
            return env_path
        else:
            print("DEBUG: cameras.txt not found in env path")
    
    possible_paths = [
        "/data/sillim_ew_mini_100024_20/sparse/0",
        "/data/samsung_dong_mini_30/sparse/0", 
        "/data/samsung_dong_mini_30_ng_28M/sparse/0",
        "/media2/4tb/aerial_photo_data/Sillim-dong/colmap_output/sparse/0",
        "output/sillim_ew_mini_100024_20/sparse/0",
        "output/samsung_dong_mini_30/sparse/0"
    ]
    
    print("DEBUG: Checking possible paths...")
    for path in possible_paths:
        print(f"DEBUG: Checking path: {path}")
        cameras_exists = os.path.exists(os.path.join(path, "cameras.txt"))
        images_exists = os.path.exists(os.path.join(path, "images.txt"))
        points_exists = os.path.exists(os.path.join(path, "points3D.txt"))
        
        print(f"DEBUG:   cameras.txt: {cameras_exists}")
        print(f"DEBUG:   images.txt: {images_exists}")
        print(f"DEBUG:   points3D.txt: {points_exists}")
        
        if cameras_exists and images_exists and points_exists:
            print(f"DEBUG: Found valid COLMAP data at: {path}")
            return path
    
    print("DEBUG: No COLMAP data found")
    return None

def main():
    """메인 실행"""
    print("COLMAP 3D Visualization Example")
    print("=" * 40)
    print("DEBUG: main() function started")
    
    # 명령행 인자 확인
    colmap_path = None
    print(f"DEBUG: sys.argv = {sys.argv}")
    if len(sys.argv) > 1:
        colmap_path = sys.argv[1]
        print(f"Using command line path: {colmap_path}")
    else:
        print("DEBUG: No command line args, calling find_colmap_data()")
        # COLMAP 데이터 경로 찾기
        colmap_path = find_colmap_data()
        print(f"DEBUG: find_colmap_data() returned: {colmap_path}")
    
    if colmap_path is None:
        print("ERROR: COLMAP 데이터를 찾을 수 없습니다!")
        print("다음 파일들이 필요합니다:")
        print("- cameras.txt")
        print("- images.txt") 
        print("- points3D.txt")
        print("\n사용법:")
        print("1. python example_colmap_viz.py /path/to/colmap/sparse/0")
        print("2. export COLMAP_PATH=/path/to/colmap/sparse/0 && python example_colmap_viz.py")
        return
    
    print(f"DEBUG: Checking if path exists: {colmap_path}")
    if not os.path.exists(colmap_path):
        print(f"ERROR: 경로가 존재하지 않습니다: {colmap_path}")
        return
    
    print(f"COLMAP 데이터 경로: {colmap_path}")
    print("DEBUG: Path exists, continuing...")
    
    # 시각화 객체 생성
    print("DEBUG: About to create COLMAPVisualizer object...")
    viz = COLMAPVisualizer(colmap_path)
    print("DEBUG: COLMAPVisualizer object created successfully")
    
    try:
        print("\n1. COLMAP 데이터 로딩...")
        print("DEBUG: Creating COLMAPVisualizer object...")
        
        print("DEBUG: Calling read_cameras_txt()...")
        viz.read_cameras_txt()
        print("DEBUG: read_cameras_txt() completed")
        
        print("DEBUG: Calling read_images_txt()...")
        viz.read_images_txt()
        print("DEBUG: read_images_txt() completed")
        
        print("DEBUG: Calling read_points3d_txt()...")
        viz.read_points3d_txt()
        print("DEBUG: read_points3d_txt() completed")
        
        # 데이터 요약 정보
        print(f"   - 카메라 수: {len(viz.cameras)}")
        print(f"   - 이미지 수: {len(viz.images)}")
        print(f"   - 3D 포인트 수: {len(viz.points3d)}")
        
        # 카메라 0의 바라보는 방향 출력
        print("\n=== CAMERA 0 ANALYSIS ===")
        first_image_id = list(viz.images.keys())[0]
        image = viz.images[first_image_id]
        
        camera_center = image['camera_center']
        R = image['R']
        # numpy는 colmap_visualizer에서 import되므로 직접 사용
        import numpy as np
        viewing_direction = R.T @ np.array([0, 0, 1])
        
        print(f"Camera 0 center: [{camera_center[0]:.6f}, {camera_center[1]:.6f}, {camera_center[2]:.6f}]")
        print(f"Camera 0 viewing direction: [{viewing_direction[0]:.6f}, {viewing_direction[1]:.6f}, {viewing_direction[2]:.6f}]")
        print(f"Z component: {viewing_direction[2]:.6f}")
        if viewing_direction[2] > 0:
            print("Camera looking UP (positive Z)")
        else:
            print("Camera looking DOWN (negative Z)")
        print("=== END CAMERA 0 ANALYSIS ===\n")
        
        print("\n2. DTM 생성...")
        # 해상도는 데이터 크기에 따라 조정
        num_points = len(viz.points3d)
        print(f"DEBUG: Number of 3D points: {num_points}")
        if num_points > 100000:
            resolution = 2.0  # 큰 데이터셋은 낮은 해상도
        elif num_points > 50000:
            resolution = 1.5
        else:
            resolution = 1.0  # 작은 데이터셋은 높은 해상도
            
        print(f"DEBUG: Using DTM resolution: {resolution}m")
        print("DEBUG: Calling create_dtm()...")
        viz.create_dtm(resolution=resolution)
        print("DEBUG: create_dtm() completed")
        print(f"   - DTM 해상도: {resolution}m")
        
        print("\n3. 3D 장면 시각화 생성...")
        print("DEBUG: Calling visualize_3d_scene()...")
        scene_center = viz.visualize_3d_scene(
            save_path='colmap_3d_scene.png'
        )
        print("DEBUG: visualize_3d_scene() completed")
        
        print(f"   - Scene Center: {scene_center}")
        
        print("\n4. Orthographic nadir view 생성...")
        print("DEBUG: Calling render_orthographic_view()...")
        viz.render_orthographic_view(scene_center, save_path='orthographic_nadir_view.png')
        print("DEBUG: render_orthographic_view() completed")
        
        print("\n✅ 모든 시각화 완료!")
        print("생성된 파일:")
        print("- colmap_3d_scene.png: 3D 장면 시각화")
        print("- orthographic_nadir_view.png: 수직 orthographic view")
        print("\n프로그램을 종료합니다 (디버깅 모드)")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("DEBUG: Exception occurred, printing traceback...")
        import traceback
        traceback.print_exc()
        
        # 디버깅 정보
        print("\n디버깅 정보:")
        print(f"- 작업 디렉토리: {os.getcwd()}")
        print(f"- COLMAP 경로: {colmap_path}")
        print(f"- 파일 존재 여부:")
        for file in ["cameras.txt", "images.txt", "points3D.txt"]:
            path = os.path.join(colmap_path, file)
            exists = os.path.exists(path)
            if exists:
                size = os.path.getsize(path)
                print(f"  - {file}: 존재 ({size:,} bytes)")
            else:
                print(f"  - {file}: 없음")
        print("DEBUG: Exception handling completed")

def test_with_sample_data():
    """샘플 데이터로 테스트 (실제 COLMAP 데이터가 없을 때)"""
    print("샘플 데이터로 테스트...")
    
    # 간단한 synthetic COLMAP 데이터 생성 (테스트용)
    import numpy as np
    
    # 임시 디렉토리 생성
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # 간단한 cameras.txt 생성
    with open(os.path.join(temp_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("1 PINHOLE 640 480 500 500 320 240\n")
    
    # 간단한 images.txt 생성  
    with open(os.path.join(temp_dir, "images.txt"), "w") as f:
        f.write("# Image list with one image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 test.jpg\n")
        f.write("# 2D points list\n")
    
    # 간단한 points3D.txt 생성
    with open(os.path.join(temp_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list:\n") 
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        np.random.seed(42)
        for i in range(1000):
            x = np.random.uniform(-50, 50)
            y = np.random.uniform(-50, 50) 
            z = np.random.uniform(0, 10)
            r, g, b = np.random.randint(0, 255, 3)
            f.write(f"{i+1} {x:.3f} {y:.3f} {z:.3f} {r} {g} {b} 0.5\n")
    
    print(f"테스트 데이터 생성: {temp_dir}")
    return temp_dir

if __name__ == "__main__":
    print("DEBUG: __main__ block started")
    print(f"DEBUG: sys.argv = {sys.argv}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("DEBUG: Running in test mode")
        # 샘플 데이터로 테스트
        temp_dir = test_with_sample_data()
        viz = COLMAPVisualizer(temp_dir)
        try:
            viz.read_cameras_txt()
            viz.read_images_txt() 
            viz.read_points3d_txt()
            viz.create_dtm(resolution=2.0)
            scene_center = viz.visualize_3d_scene()
            viz.render_orthographic_view(scene_center)
            print("테스트 완료!")
        except Exception as e:
            print(f"테스트 실패: {e}")
        finally:
            # 임시 파일 정리
            import shutil
            shutil.rmtree(temp_dir)
    else:
        print("DEBUG: Running in normal mode, calling main()")
        main()
        print("DEBUG: main() function completed")
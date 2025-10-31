# Progressive Training Resume 기능

## 개요
Progressive Training에서 특정 윈도우에서 중단되었을 때, 해당 윈도우부터 다시 시작할 수 있는 resume 기능이 추가되었습니다.

## 사용 방법

### 1. usage_progressive.sh에서 설정
```bash
# usage_progressive.sh 파일에서 다음 라인을 수정:
RESUME_FROM_WINDOW="auto"    # 자동 감지 (권장)
# 또는
RESUME_FROM_WINDOW="5"       # 특정 윈도우 번호
```

### 2. 직접 명령어로 실행
```bash
python3 progressive_learning/run_progressive.py \
  --source_path="/data/your_dataset" \
  --output_path="./output/your_output" \
  --resume_from_window="auto" \
  [other parameters...]
```

## Resume 옵션 설명

### `auto` (권장)
- 자동으로 마지막 완료된 윈도우를 감지
- 다음 윈도우부터 자동으로 재시작
- 가장 안전하고 편리한 방법

### 특정 윈도우 번호
- `0`: initial 윈도우부터 재시작
- `1`: window_001부터 재시작
- `2`: window_002부터 재시작
- 등등...

### `""` (빈 문자열)
- Resume 기능 비활성화 (기본값)
- 처음부터 새로 시작

## 작동 원리

1. **Auto-detection**:
   - `output/your_output/model_*` 디렉토리들을 검사
   - 각 디렉토리의 `checkpoints/` 폴더 존재 여부로 완료 상태 확인
   - 마지막으로 완료된 윈도우 다음부터 재시작

2. **State restoration**:
   - 이전 윈도우의 `state.json` 파일에서 카메라 정보 복원
   - 학습 진행 상태 복원
   - Checkpoint에서 모델 상태 복원

3. **Safety checks**:
   - Output 디렉토리 존재 여부 확인
   - State 파일 및 checkpoint 존재 여부 확인
   - 실패 시 처음부터 새로 시작

## 사용 예시

### 예시 1: 자동 감지로 재시작
```bash
# 학습이 window_005에서 중단된 경우
cd /workspace/Grendel-GS
./usage_progressive.sh your_dataset

# usage_progressive.sh에서 다음과 같이 설정:
RESUME_FROM_WINDOW="auto"
```

### 예시 2: 특정 윈도우부터 재시작
```bash
# window_003부터 강제로 재시작하려는 경우
RESUME_FROM_WINDOW="3"
```

## 주의사항

1. **기존 output 디렉토리 보존**: Resume 모드에서는 기존 output 디렉토리를 삭제하지 않습니다.

2. **State 파일 의존성**: 이전 윈도우의 `state.json` 파일이 필요합니다.

3. **Checkpoint 의존성**: 이전 윈도우의 checkpoint 파일들이 필요합니다.

4. **설정 파일**: `convergence_config.txt`는 resume 시에도 다시 로드됩니다.

5. **Trajectory 정보**: `balanced_smooth_trajectory` 전략 사용 시 `trajectory_info.json` 파일이 생성됩니다.

## 에러 처리

### Auto mode (`"auto"`)
Resume 실패 시 자동으로 처음부터 새로 시작합니다:
- 완료된 윈도우가 없는 경우
- State 파일 복원 실패 시

### 특정 윈도우 mode (`"N"`)
Resume 실패 시 **프로그램이 종료됩니다**:
- State 파일이 없는 경우
- Checkpoint 디렉토리가 없는 경우
- Checkpoint 파일이 없는 경우
- State 파일이 손상된 경우
- Dataset 불일치가 감지된 경우
- 기타 복원 실패 시

**사용자가 직접 결정해서 다시 실행해야 합니다:**
```bash
# Resume 실패 시 선택지:
# 1. 다른 윈도우 번호로 재시도
RESUME_FROM_WINDOW="3"

# 2. Auto 모드로 재시도
RESUME_FROM_WINDOW="auto"

# 3. 처음부터 새로 시작
RESUME_FROM_WINDOW=""
```

## 로그 메시지

Resume 시 다음과 같은 메시지들을 확인할 수 있습니다:
```
🔄 Resume mode: Using existing output directory output/progressive_test
🔍 Auto-detected last completed window: 5
   Will resume from window 6
🔄 Restoring from window 5...
✅ Dataset compatibility verified: /data/Samsung_SN_30
✅ Trajectory info restored from: output/progressive_test/model_window_005/trajectory_info.json
   Window centers: 6
   Camera history: [1, 3, 7, 12, 15]
✅ State restored:
   Window cameras: [1, 5, 8, 12, 15]
   Iteration count: 5
   Checkpoint files: 8 found
🔄 Resumed from window 6 at iteration 5
```

## 🆕 Trajectory 정보 복원

`balanced_smooth_trajectory` 전략 사용 시:
- **자동 저장**: 각 윈도우 완료 후 `trajectory_info.json` 생성
- **자동 복원**: Resume 시 trajectory 연속성 유지
- **호환성**: 기존 state.json과 독립적으로 작동
- **실패 시 대응**: Trajectory 파일이 없어도 기본 resume은 가능
# Double Dict Live Test - Ten Model Test

## 개요
PyTorch 기반 Double Dict Live Test 시스템입니다. 10개의 모델을 실시간으로 변경하면서 모델 추론을 수행하고, 모델 변경 시간을 측정합니다.

## 주요 특징

### 1. 동적 모델 변경
- `model0.pth` ~ `model9.pth`: 10개의 완전한 모델 파일
- 1.5초마다 자동으로 모델 변경
- 기존 모델 메모리 정리 후 새 모델 로드
- `torch.cuda.empty_cache()` 를 통한 GPU 메모리 최적화

### 2. 성능 측정
- 모델 변경 시간 실시간 측정
- 통계 정보 제공:
  - 총 변경 횟수
  - 평균 변경 시간
  - 최소/최대 변경 시간

### 3. Live Test 기능
- 실시간 모델 스위칭
- GPU 메모리 사용량 모니터링
- 추론 시간 측정
- 결과에 현재 사용 중인 모델 정보 포함
- 사용 가능한 모델 목록 제공

## 파일 구조
```
Ten Model Test/
├── model_handler.py     # Live Test 모델 핸들러
├── model0.pth          # 모델 #0
├── model1.pth          # 모델 #1
├── model2.pth          # 모델 #2
├── model3.pth          # 모델 #3
├── model4.pth          # 모델 #4
├── model5.pth          # 모델 #5
├── model6.pth          # 모델 #6
├── model7.pth          # 모델 #7
├── model8.pth          # 모델 #8
├── model9.pth          # 모델 #9
├── label.txt           # 클래스 라벨
└── README.md           # 이 파일
```

## 사용 방법

### 1. 직접 실행
```bash
cd "PyTorch/Double Dict Live Test/Ten Model Test"
python model_handler.py
```

### 2. 프로그래밍 방식
```python
import pickle
import cv2
import numpy as np
from model_handler import ModelHandler

# 이미지 준비
img = cv2.imread('test_image.jpg')
img = cv2.resize(img, (299, 299))
data = pickle.dumps(img)

# Live Test 실행
handler = ModelHandler(None, None)
result, context = handler(data=data, context={})

# 결과 확인
result_dict = pickle.loads(result)
print(f"결과: {result_dict['result_code']}")
print(f"점수: {result_dict['score']:.4f}")
print(f"현재 모델: {result_dict['current_model']}")
print(f"변경 통계: {result_dict['model_change_stats']}")
print(f"사용 가능한 모델: {result_dict['available_models']}")
```

## 로그 출력 예시
```
[LOG] 10개 모델 파일 확인 중...
[LOG] model0.pth 발견
[LOG] model1.pth 발견
[LOG] model2.pth 발견
...
[LOG] 총 10개 모델 파일 발견
[LOG] Live Test 모드 초기화 완료
[MODEL CHANGE] 모델 변경 시작...
[MODEL CHANGE] model3.pth 로드 완료
[MODEL CHANGE TIME] 모델 변경 시간: 0.1234초
[MODEL CHANGE STATS] 총 변경: 1회, 평균: 0.1234초, 최소: 0.1234초, 최대: 0.1234초
[CURRENT MODEL] 현재 사용 중인 모델: model3.pth
[PREDICT TIME] 추론 시간: 0.0145초
```

## 성능 특징
- 모델 변경 시간: 보통 0.1-0.2초
- 모델 간 자동 전환: 1.5초 간격
- GPU 메모리 효율적 관리
- 실시간 통계 업데이트
- 부분 모델 지원 (일부 모델 파일이 없어도 동작)

## 모델 파일 관리

### 1. 모델 파일 확인
시스템은 자동으로 `model0.pth` ~ `model9.pth` 파일을 확인합니다:
```python
for i in range(10):
    model_file = f"model{i}.pth"
    if os.path.exists(model_path):
        self.model_files.append(model_file)
```

### 2. 동적 모델 변경
```python
# 기존 모델 정리
if self.current_model is not None:
    del self.current_model
    torch.cuda.empty_cache()

# 새 모델 로드
self.current_model = torch.load(model_path, map_location=self.device)
self.current_model.to(self.device)
self.current_model.eval()
```

### 3. 메모리 최적화
- 기존 모델 객체 즉시 삭제
- GPU 캐시 정리
- 메모리 사용량 실시간 모니터링

## 주의사항
- GPU 메모리 제한: 300MB (설정 가능)
- 입력 이미지 크기: 299x299
- PyTorch 및 CUDA 환경 필요
- 최소 1개 이상의 모델 파일 필요
- 모든 모델 파일이 동일한 입력/출력 구조를 가져야 함

## 확장성

### 1. 모델 수 변경
모델 수를 변경하려면 다음을 수정하세요:
```python
# 10개 대신 다른 수로 변경
for i in range(20):  # 20개 모델로 확장
    model_file = f"model{i}.pth"
```

### 2. 변경 간격 조정
```python
self.model_change_interval = 2.0  # 2초로 변경
```

### 3. 메모리 제한 변경
```python
memory_limit = 512  # 512MB로 변경
```

## 기술적 세부사항
- **모델 변경 방식**: 완전한 모델 파일 교체
- **메모리 관리**: 기존 모델 삭제 후 새 모델 로드
- **시간 측정**: `time.time()` 기반 정밀 측정
- **통계 누적**: 실행 중 지속적으로 업데이트
- **GPU 최적화**: `torch.cuda.empty_cache()` 사용

## 트러블슈팅

### 1. 모델 파일 없음
```
[WARNING] model3.pth 파일이 없습니다.
```
→ 해당 모델 파일을 추가하거나 무시하고 진행

### 2. GPU 메모리 부족
```
[GPU MEMORY ERROR] CUDA out of memory
```
→ `memory_limit` 값을 줄이거나 모델 크기 확인

### 3. 모델 호환성 문제
```
[MODEL CHANGE ERROR] 모델 변경 실패
```
→ 모든 모델 파일이 동일한 구조인지 확인 
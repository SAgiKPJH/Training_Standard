# Double Dict Live Test - Two Model Test

## 개요
PyTorch 기반 Double Dict Live Test 시스템입니다. 실시간으로 가중치를 변경하면서 모델 추론을 수행하고, 가중치 변경 시간을 측정합니다.

## 주요 특징

### 1. 동적 가중치 변경
- `model.pth`: 기본 모델 구조 + 원본 가중치
- `weight2.pth`: 추가 가중치 (state_dict)
- 2초마다 자동으로 가중치 변경
- `copy.deepcopy()` + `load_state_dict()` 방식 사용

### 2. 성능 측정
- 가중치 변경 시간 실시간 측정
- 통계 정보 제공:
  - 총 변경 횟수
  - 평균 변경 시간
  - 최소/최대 변경 시간

### 3. Live Test 기능
- 실시간 가중치 스위칭
- GPU 메모리 사용량 모니터링
- 추론 시간 측정
- 결과에 현재 사용 중인 가중치 정보 포함

## 파일 구조
```
Two Model Test/
├── model_handler.py     # Live Test 모델 핸들러
├── model.pth           # 기본 모델 (구조 + 가중치)
├── weight2.pth         # 추가 가중치 (state_dict)
├── label.txt           # 클래스 라벨
└── README.md           # 이 파일
```

## 사용 방법

### 1. 직접 실행
```bash
cd "PyTorch/Double Dict Live Test/Two Model Test"
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
print(f"현재 가중치: {result_dict['current_weight']}")
print(f"변경 통계: {result_dict['weight_change_stats']}")
```

## 로그 출력 예시
```
[LOG] 기본 모델 구조 로드 성공 (model.pth)
[LOG] Live Test 모드 초기화 완료
[WEIGHT CHANGE] 가중치 변경 시작...
[WEIGHT CHANGE] weight2.pth 가중치 로드 완료
[WEIGHT CHANGE TIME] 가중치 변경 시간: 0.0234초
[WEIGHT CHANGE STATS] 총 변경: 1회, 평균: 0.0234초, 최소: 0.0234초, 최대: 0.0234초
[CURRENT MODEL] 현재 사용 중인 가중치: weight2.pth
[PREDICT TIME] 추론 시간: 0.0145초
```

## 성능 특징
- 가중치 변경 시간: 보통 0.02-0.05초
- 모델 간 자동 전환: 2초 간격
- GPU 메모리 효율적 관리
- 실시간 통계 업데이트

## 주의사항
- GPU 메모리 제한: 300MB (설정 가능)
- 입력 이미지 크기: 299x299
- PyTorch 및 CUDA 환경 필요
- 가중치 파일이 모델 구조와 호환되어야 함

## 기술적 세부사항
- **가중치 변경 방식**: `copy.deepcopy(base_model)` 후 `load_state_dict()`
- **메모리 관리**: 사용하지 않는 모델 객체 즉시 정리
- **시간 측정**: `time.time()` 기반 정밀 측정
- **통계 누적**: 실행 중 지속적으로 업데이트 
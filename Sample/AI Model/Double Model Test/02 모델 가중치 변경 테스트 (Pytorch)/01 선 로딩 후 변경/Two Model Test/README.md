# PyTorch Double Dict Test - Two Model Test

## 개요
분리된 모델 파일들을 사용하여 두 개의 가중치 세트를 시간 기반으로 전환하는 PyTorch 모델 추론 시스템입니다.

## 파일 구조
```
Two Model Test/
├── model_handler.py          # 메인 모델 핸들러
├── label.txt                 # 분류 라벨 (9개 클래스)
├── architecture_info.json    # 모델 아키텍처 정보
├── model.pth                 # 모델 구조 파일 (사용자 추가 필요)
├── weight.pth                # 첫 번째 가중치 파일 (사용자 추가 필요)
├── weight2.pth               # 두 번째 가중치 파일 (사용자 추가 필요)
└── README.md                 # 이 파일
```

## 핵심 기능

### 1. 분리된 모델 로딩
- **model.pth**: 실제 모델 구조 (PyTorch 모델 객체)
- **weight.pth**: 첫 번째 가중치 세트
- **weight2.pth**: 두 번째 가중치 세트
- **architecture_info.json**: 모델 메타데이터

### 2. 시간 기반 모델 선택
```python
decimal_part = time.time() - int(time.time())
selected_model = int(decimal_part * 100) % 2
```
- 짝수: weight2.pth 사용
- 홀수: weight.pth 사용

### 3. 클래스 수 자동 감지
- `label.txt`에서 클래스 수를 자동으로 읽어옴
- 형식: `01,02,03,04,05,06,07,08,09`

### 4. GPU 메모리 모니터링
- 듀얼 GPU 지원 (GPU 0, 1)
- 메모리 사용량 실시간 모니터링
- 로딩 단계별 메모리 추적

## 사용법

### 1. 필요 파일 준비
다음 파일들을 사용자가 직접 추가해야 합니다:
- `model.pth`: 실제 모델 구조
- `weight.pth`: 첫 번째 가중치
- `weight2.pth`: 두 번째 가중치

### 2. 모델 실행
```python
from model_handler import ModelHandler
import cv2
import pickle

# 이미지 로드
img = cv2.imread('path/to/image.jpg')
img = cv2.resize(img, (299, 299))
data = pickle.dumps(img)

# 모델 추론
handler = ModelHandler(None, None)
result, context = handler(data, {})
```

### 3. 결과 확인
```python
import pickle
result_dict = pickle.loads(result)
print(f"분류 결과: {result_dict['result_code']}")
print(f"신뢰도: {result_dict['score']:.4f}")
```

## 기술적 특징

### 모델 구조 복사
```python
import copy
model = copy.deepcopy(self.base_model)
model.load_state_dict(weight_dict, strict=False)
```

### 호환성 검사
- 가중치 레이어 이름과 형태 자동 검증
- 호환 가능한 가중치만 선택적 로드
- 오류 발생 시 안전한 폴백 처리

### 메모리 최적화
- CPU에서 가중치 로드 후 GPU로 전송
- 불필요한 메모리 사용 최소화
- 메모리 사용량 실시간 모니터링

## 로깅 정보

### 모델 로딩 과정
```
[LOG] 모델 구조 로드 완료: model.pth
[LOG] 가중치 로드 완료: weight.pth
[LOG] label.txt에서 클래스 수 확인: 9개
[LOG] 모델 구조 복사 완료: InceptionV3
[LOG] 가중치 로드 성공: 377개 레이어
```

### 메모리 사용량
```
[GPU0 MEMORY 모델 로딩 전] 사용: 0.0MB, 여유: 300.0MB
[GPU0 MEMORY 첫 번째 가중치 로딩 후] 사용: 95.3MB, 여유: 204.7MB
[GPU0 MEMORY 두 번째 가중치 로딩 후] 사용: 190.6MB, 여유: 109.4MB
```

### 추론 과정
```
[LOG] 소수점 이하: 0.7234 -> 72 % 2 = 0 -> weight2.pth 사용
[LOG] 모델 출력 형태: (1, 9)
[LOG] 최종 예측 결과: 클래스=05, 신뢰도=0.8234
```

## 주의사항

1. **필수 파일**: model.pth, weight.pth, weight2.pth는 사용자가 직접 준비해야 합니다.
2. **호환성**: 가중치 파일이 모델 구조와 호환되어야 합니다.
3. **메모리**: 두 개의 모델을 동시에 로드하므로 충분한 GPU 메모리가 필요합니다.
4. **이미지 크기**: 입력 이미지는 299x299로 자동 리사이즈됩니다.

## 오류 처리

### 파일 없음
```python
if self.base_model is None:
    raise ValueError("모델 구조 로드에 실패했습니다. model.pth에 실제 모델 객체가 있어야 합니다.")
```

### 가중치 호환성
```python
if compatible_dict:
    model.load_state_dict(compatible_dict, strict=False)
    logger.info(f"호환 가능한 가중치 {len(compatible_dict)}/{len(weight_dict)}개 로드")
```

### 라벨 파일 오류
```python
except:
    num_classes = 9  # 기본값
    logger.info("label.txt 읽기 실패, 기본값 사용: 9개")
```

이 시스템은 실제 모델 파일이 준비되었을 때 완전히 작동합니다. 
# ConvNeXt Inference Weight (가중치 교체 방식)

이 폴더는 **하나의 ConvNeXt 모델**에서 **가중치만 교체**하여 두 가지 다른 모델을 사용하는 방식입니다.

## 특징

### ⚡ **효율적인 가중치 교체**
- 하나의 모델 구조만 메모리에 유지
- 가중치만 동적으로 교체 (`load_state_dict` 사용)
- 메모리 효율적이면서도 빠른 전환
- **Mixed Precision 호환**: float32/float16 자동 처리

### 🔄 **가중치 관리**
- `model.pth`: 기본 가중치 (초기 로딩)
- `weight.pth`: 대체 가중치 (필요시 교체)
- 시간 기반 자동 선택 (밀리초 % 2)

### 📊 **성능 최적화**
- 불필요한 가중치 로딩 방지 (동일 타입 스킵)
- 가중치 변경 시간 실시간 측정 및 통계
- GPU 메모리 사용량 모니터링

## 파일 구조

```
convnext_inference_weight/
├── model_handler.py    # 가중치 교체 방식 모델 핸들러
├── label.txt          # 클래스 라벨 파일
├── model.pth          # 기본 가중치 (추가 필요)
├── weight.pth         # 대체 가중치 (추가 필요)
└── README.md          # 이 파일
```

## 작동 원리

### 1. 초기화
```python
# 1. 단일 ConvNeXt 모델 생성
self.model = create_model('convnext_large_384_in22ft1k', ...)

# 2. 기본 가중치 로드
self.load_weights('model')  # model.pth 로드
```

### 2. 추론 시 가중치 선택
```python
# 시간 기반 가중치 타입 결정
target_weight_type = self.determine_weight_type()  # 'model' or 'weight'

# 필요시에만 가중치 교체
if current_weight_type != target_weight_type:
    self.model.load_state_dict(checkpoint['model_state_dict'])
```

## 사용법

### 1. 가중치 파일 준비
```bash
# 가중치 파일들을 이 폴더에 복사
cp your_base_model.pth ./model.pth
cp your_alternative_weights.pth ./weight.pth
```

### 2. 실행
```python
from model_handler import ModelHandler
import pickle
import numpy as np

# 더미 이미지 데이터 (384x384x3)
dummy_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
data = pickle.dumps(dummy_image)

# 모델 핸들러 초기화 (기본 가중치로 시작)
handler = ModelHandler(None, None)

# 추론 실행
result, context = handler(data=data, context={})
result_dict = pickle.loads(result)

print(f"결과: {result_dict['result_code']}")
print(f"점수: {result_dict['score']}")
print(f"사용된 가중치: {result_dict['current_weight']}")
print(f"가중치 변경 통계: {result_dict['weight_change_stats']}")
```

## 성능 비교

| 방식 | 초기화 시간 | 가중치 변경 | 메모리 사용량 | 전환 속도 |
|------|-------------|-------------|---------------|-----------|
| **가중치 교체** | 빠름 (한 모델) | **중간** (~1-2초) | **낮음** (한 모델) | 중간 |
| 사전 로딩 | 느림 (두 모델) | 매우 빠름 (~0.001초) | 높음 (두 모델) | 매우 빠름 |
| 동적 로딩 | 빠름 (한 모델) | 느림 (~6초) | 낮음 (한 모델) | 느림 |

## 로그 예시

```
[MODEL INIT] ConvNeXt 모델 생성 시작...
[WEIGHT LOAD] 가중치 로드 시작: model
[WEIGHT LOAD] model.pth 가중치 로드 완료
[MODEL INIT] 초기 가중치 로드 완료 (model.pth)

[WEIGHT SELECT] 시간: 1754284103.164, 밀리초: 163, %2 = 1 -> weight
[WEIGHT CHANGE] 가중치 변경 시작: model -> weight
[WEIGHT LOAD] 가중치 로드 시작: weight
[WEIGHT CHANGE TIME] 가중치 변경 시간: 1.2347초
[WEIGHT LOAD] weight.pth 가중치 로드 완료
[CURRENT MODEL] 현재 사용 중인 가중치: weight
[PREDICT TIME] 추론 시간: 0.0234초
```

## 장점

✅ **메모리 효율성**: 하나의 모델만 메모리에 유지  
✅ **적당한 속도**: 가중치 교체 시간이 전체 모델 교체보다 빠름  
✅ **유연성**: 다양한 가중치를 쉽게 추가 가능  
✅ **안정성**: 가중치 로드 실패시 현재 가중치 유지  

## 단점

⚠️ **전환 지연**: 사전 로딩 방식보다는 느림  
⚠️ **동일 구조**: 같은 모델 아키텍처의 가중치만 사용 가능  

## 권장 사용 시나리오

- **메모리 제약**: GPU 메모리가 제한적인 환경
- **중간 성능**: 속도와 메모리 사이의 균형이 필요한 경우  
- **가중치 실험**: 다양한 가중치를 테스트하는 환경
- **배치 처리**: 적당한 빈도의 모델 전환이 필요한 경우
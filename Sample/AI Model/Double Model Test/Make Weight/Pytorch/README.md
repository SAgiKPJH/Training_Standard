# PyTorch Weight Extractor

PyTorch 모델에서 가중치만 추출하는 간단한 도구입니다.

## 주요 기능

- **가중치 추출**: 모델에서 state_dict만 추출
- **간단한 사용법**: 모델 로드 → 가중치 저장
- **다양한 형태 지원**: 모델 객체, state_dict 모두 지원

## 사용 방법

### 기본 사용법
```bash
# 가중치 추출
python model_weight_separator.py -i input_model.pth -o output_directory
```

### 결과
```
output_directory/
└── weights.pth     # 가중치 정보 (state_dict)
```

## 추출된 가중치 사용 방법

```python
import torch

# 가중치 로드
state_dict = torch.load('output_directory/weights.pth')

# 새로운 모델에 적용
model = YourModel()  # 동일한 구조의 모델 생성 필요
model.load_state_dict(state_dict)
model.eval()
```

## 예제

### 1. 가중치 추출
```bash
python model_weight_separator.py -i my_model.pth -o extracted_weights
```

### 2. 추출된 가중치 사용
```python
import torch

# 가중치 로드
weights = torch.load('extracted_weights/weights.pth')

# 모델 생성 (동일한 구조)
model = create_your_model()

# 가중치 적용
model.load_state_dict(weights)
model.eval()

# 추론 가능
output = model(input_data)
```

## 지원 형태

- **모델 객체**: `torch.save(model, 'model.pth')`로 저장된 파일
- **State Dict**: `torch.save(model.state_dict(), 'weights.pth')`로 저장된 파일
- **TorchScript**: `torch.jit.save(model, 'model.pt')`로 저장된 파일

## 주의사항

1. **모델 구조**: 가중치만 추출되므로 사용 시 동일한 구조의 모델이 필요합니다
2. **호환성**: 추출된 가중치는 원본 모델과 동일한 구조에서만 사용 가능합니다
3. **백업**: 원본 모델 파일을 백업하세요

## 장점

- **간단함**: 복잡한 구조 분리 없이 가중치만 추출
- **안전함**: 원본 모델을 변경하지 않음
- **빠름**: 단순한 작업으로 빠른 처리

## 사용 사례

- 모델 가중치 백업
- 다른 모델로 가중치 전이
- 모델 크기 줄이기 (구조 코드 별도 관리)
- 가중치 분석 및 시각화

이 도구는 **가중치만 추출**하는 단순한 기능에 집중합니다. 
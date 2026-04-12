---
name: add-model
description: 새 모델 추가 시 체크리스트
---

## 새 모델 추가 절차

### 1. Pytorch_Classification_Models.py
- `_TORCHVISION_MODELS`에 모델명 추가 (문자열 매핑)
- 특수 파라미터 필요 시 `init_model`에 조건 분기 추가
- `getattr(torchvision.models, name, None)` 방식이므로 구버전에서 없어도 에러 안남

### 2. Tensorflow_Classification_Models.py
- `__network_map`에 `tf.keras.applications.*` 추가

### 3. ui.json (3개)
- Step1의 Options에 모델 항목 추가
- DefaultValues로 해당 모델의 권장 input_size 설정
- CNN_Training_Standard: PyTorch/TensorFlow 각각의 Condition 블록에 추가

### 4. 주의사항
- `__init__`에서 `torchvision.models.xxx` 직접 참조 금지 (구버전 호환)
- InceptionV3처럼 특수 파라미터(aux_logits 등) 필요한 모델은 init_model에서 분기
- tuple 출력 모델: TrainingBuilder의 `isinstance(outputs, tuple)` 처리 확인
- torch.jit.script 불가 모델: trace fallback 동작 확인

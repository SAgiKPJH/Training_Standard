---
name: debug-training
description: 학습 문제 디버깅 가이드
---

## 증상별 진단

### 모든 이미지에서 동일 클래스 1.0 출력
1. Training과 Inference의 전처리 비교 (채널 순서, Resize/ToTensor 순서)
2. BGR/RGB 불일치 확인 → BGR 유지가 맞음
3. 모델 저장/로드 문제: zeros/ones/randn 입력으로 diff 확인
4. diff > 0이면 모델은 정상 → 학습 데이터 불균형 또는 과적합

### float16 overflow (NaN/Inf loss)
- 증상: `max=inf`, `dtype=torch.float16`
- 원인: AMP 사용 시 GradScaler 누락
- 해결: `scaler.scale(loss).backward()` + `scaler.step()` + `scaler.update()`

### EfficientNet has no attribute 'outputs'
- 원인: PyTorch 모델을 .h5(Keras)로 저장 시도
- 해결: `hasattr(model, 'state_dict')` 체크 후 .h5 → .pth 변환

### module has no attribute 'GradScaler'
- 원인: torch 버전 차이
- 해결: `except (TypeError, AttributeError)` → cuda.amp fallback

### ZeroDivisionError (save_epoch)
- 원인: save_epoch = 0
- 해결: `save_epoch > 0 and epoch % save_epoch == 0` 가드

### Inception3 has no attribute 'outputs' (저장 시)
- 원인: torch.jit.script가 InceptionV3에서 실패
- 해결: script → trace → state_dict fallback 체인

### Debug 모드
- parameter에 `"debug": true` 설정
- epoch 20부터 상세 로그: per-sample loss, logit 분포, gradient, weight update
- loss spike 감지 (3σ 초과 시 WARNING)

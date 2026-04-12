---
name: training-standard
description: Training Standard 프로젝트의 핵심 규칙과 패턴
---

## 이미지 채널
- mpp.intel64.load, cv2.imread 모두 BGR로 로드
- Image.fromarray()는 채널 swap 없이 BGR 그대로 전달
- **학습이 BGR로 됨 → Inference에서 BGR2RGB 변환 금지**
- 전처리 불일치 시 모델이 항상 동일 클래스 1.0 출력하는 증상 발생

## mpp 의존성
- DAQ_* 모듈은 mpp 패키지 필요 (서버 환경)
- Local_* 모듈은 mpp 없이 동작
- __init__.py에서 DAQ 모듈은 `__getattr__` lazy import 필수
- 직접 import 시 로컬에서 ModuleNotFoundError

## 모델 관련
- network_name: 기본값 None, 미지정 시 즉시 에러 (캐시 문제 방지)
- 모델 맵: `getattr(torchvision.models, name, None)` (구버전 호환)
- InceptionV3: aux_logits=False 고정
- 모델 저장: jit.script → jit.trace → state_dict fallback 순서

## torch 버전 호환
```python
# GradScaler
try:
    scaler = torch.amp.GradScaler('cuda', enabled=using_amp)
except (TypeError, AttributeError):
    scaler = torch.cuda.amp.GradScaler(enabled=using_amp)
```
- torch.amp.autocast vs torch.cuda.amp.autocast 둘 다 지원
- torchvision에 없는 모델은 getattr로 런타임 확인

## save_epoch
- 0: 마지막 epoch에만 저장
- `epoch % save_epoch` 전에 반드시 `save_epoch > 0` 체크 (ZeroDivisionError 방지)

## daq_old_path
- true일 때 기존 train_recipe.py 저장 경로 패턴 사용
- epoch_{N}/model/model.h5 형식
- PyTorch 모델이 .h5 경로로 저장 요청되면 자동으로 .pth 변환
- `hasattr(model, 'state_dict')` 체크

## best 모델
- validation_loss 기준 단일 저장 (best/model.pth)
- validation 없으면 train_loss 기준 fallback

## temp 폴더
- 실행 py 위치에 temp/{uuid}/ 생성
- `sys.modules['__main__'].__file__` 기준
- 학습 완료 후 shutil.rmtree로 삭제

## Training ↔ Inference 전처리 일치
반드시 동일해야 하는 것:
1. 채널 순서 (BGR)
2. Resize → ToTensor 순서 (ToTensor 먼저 하면 보간법 차이)
3. Normalize mean/std 값

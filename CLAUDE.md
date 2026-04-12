# Training Standard

CNN Classification 학습/추론 표준화 프레임워크. PyTorch/TensorFlow 지원.

## 구조
- `Training Standard/Pytorch_CNN_Standard/` - PyTorch 전용
- `Training Standard/Tensorflow_CNN_Standard/` - TensorFlow 전용
- `Training Standard/CNN_Training_Standard/` - 통합 (framework 파라미터 선택)
- `Inference Standard/` - 추론
- `create_dataset/` - 샘플 데이터 생성

## Builder 패턴
`Builder/` 하위: ClassCode, Dataset, HyperParameter, Model, Monitoring, Operation, Save, Train
- Local_* : mpp 없이 동작 / DAQ_* : mpp 필요 (서버), lazy import

## 핵심 규칙
- **BGR 유지**: 학습/추론 모두 BGR (RGB 변환 금지)
- **mpp lazy import**: DAQ 모듈은 `__getattr__`로 지연 로드
- **network_name 필수**: 기본값 None → 미지정 시 즉시 에러
- **모델 저장**: jit.script → jit.trace → state_dict fallback
- **GradScaler**: `try torch.amp / except (TypeError, AttributeError) → torch.cuda.amp`
- **모델 맵**: `getattr(torchvision.models, name, None)` (버전 호환)
- **InceptionV3**: `aux_logits=False` 고정
- **save_epoch 0**: 마지막만 저장 (ZeroDivision 방지)
- **daq_old_path**: PyTorch `.h5` → `.pth` 자동 변환
- **temp 폴더**: 실행 py 위치에 생성, 학습 후 삭제

## 코드 품질 검사 (Hooks)
파일 저장(Write/Edit) 시 자동 실행:
- **ERROR**: py 문법 오류 → 블로킹 (저장 차단)
- **DANGER**: py 200줄 초과 → 분리 권장
- **WARNING**: py 150줄 초과 → 주의

전체 스캔: `python .claude/hooks/scan_all.py`
DANGER/WARNING 발생 시 코드 분리를 검토하고 개선할 것.

## 파라미터 JSON
```json
{"version":"..._v1.0.3","hyperparameter":{"framework":"pytorch","network_name":"efficientnet_b0","input_size":224,"criterion":"CrossEntropyLoss","augmentation":{}}}
```

## UI
`ui.json`으로 분리. 네트워크별 DefaultValues로 input_size 자동 설정.

## 커밋
한국어, 기능 중심. 예: `refactor: best 두번 저장 개선`

# Training Standard

CNN Classification 모델 학습/추론 표준화 프레임워크

## 개요

PyTorch와 TensorFlow를 지원하는 CNN 분류 모델 학습 파이프라인입니다.  
Builder 패턴으로 모듈화되어 있으며, DAQ 플랫폼 및 로컬 환경 모두 지원합니다.

## 디렉토리

| 디렉토리 | 설명 |
|----------|------|
| `Training Standard/Pytorch_CNN_Standard` | PyTorch 전용 학습 |
| `Training Standard/Tensorflow_CNN_Standard` | TensorFlow 전용 학습 |
| `Training Standard/CNN_Training_Standard` | 통합 학습 (framework 선택) |
| `Inference Standard/` | 모델 추론 (Pytorch/Tensorflow) |
| `create_dataset/` | 샘플 데이터셋 생성기 |
| `Sample/` | 참고 샘플 코드 |

## 빠른 시작

### 1. 샘플 데이터 생성
```bash
cd create_dataset
pip install -r requirements.txt
python create_sample_dataset.py --output ./dataset --count 50
```

### 2. 로컬 학습 (PyTorch)
```bash
cd "Training Standard/Pytorch_CNN_Standard"
python pytorch_classification_local.py
```

### 3. 추론
```bash
cd "Inference Standard/Pytorch_CNN_Standard"
python model_handler.py
```

## 파라미터

JSON에서 하이퍼파라미터를 설정합니다:

```json
{
  "version": "Pytorch_CNN_Standard_v1.0.3",
  "hyperparameter": {
    "network_name": "efficientnet_b0",
    "epoch": 20,
    "batch_size": 16,
    "lr": 0.001,
    "input_size": 224,
    "using_gpu": true,
    "criterion": "CrossEntropyLoss",
    "augmentation": {
      "horizontal_flip": true,
      "rotation": 15
    }
  }
}
```

## 코드 품질

```bash
python .claude/hooks/scan_all.py
```
- py 200줄 초과: DANGER (분리 권장)
- py 150줄 초과: WARNING
- 문법 오류: ERROR

## 버전
- v1.0.3 (현재)

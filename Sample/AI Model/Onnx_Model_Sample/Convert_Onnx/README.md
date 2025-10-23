# ONNX 모델 변환 도구

이 폴더는 다양한 딥러닝 프레임워크 간 모델 변환을 위한 스크립트들을 포함합니다.

## 지원하는 변환

1. **PyTorch → ONNX**: `pytorch_to_onnx.py`
2. **TensorFlow → ONNX**: `tensorflow_to_onnx.py`
3. **ONNX → PyTorch**: `onnx_to_pytorch.py`
4. **ONNX → TensorFlow**: `onnx_to_tensorflow.py`

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. PyTorch → ONNX 변환

```bash
# 기본 변환
python pytorch_to_onnx.py --model_path model.pth --onnx_path model.onnx

# 검증 포함
python pytorch_to_onnx.py --model_path model.pth --onnx_path model.onnx --verify

# 커스텀 입력 형태
python pytorch_to_onnx.py --model_path model.pth --onnx_path model.onnx --input_shape 1,3,224,224
```

### 2. TensorFlow → ONNX 변환

```bash
# SavedModel 변환
python tensorflow_to_onnx.py --model_path saved_model_dir --onnx_path model.onnx

# H5 모델 변환
python tensorflow_to_onnx.py --model_path model.h5 --onnx_path model.onnx --verify
```

### 3. ONNX → PyTorch 변환

```bash
python onnx_to_pytorch.py --onnx_path model.onnx --pytorch_path model.pth --verify
```

### 4. ONNX → TensorFlow 변환

```bash
python onnx_to_tensorflow.py --onnx_path model.onnx --tf_output_dir tf_model --verify
```

## 주요 특징

- **자동 검증**: 변환 후 원본 모델과 출력 비교
- **유연한 입력 형태**: 다양한 입력 텐서 형태 지원
- **에러 처리**: 변환 실패 시 상세한 에러 메시지
- **샘플 모델**: 테스트용 샘플 모델 자동 생성

## 주의사항

1. **메모리**: 대용량 모델 변환 시 충분한 메모리 확보
2. **호환성**: 일부 복잡한 레이어는 변환되지 않을 수 있음
3. **버전**: 프레임워크 버전 호환성 확인 필요

## 문제 해결

### 일반적인 오류

1. **ImportError**: 필요한 패키지 설치 확인
2. **Shape Mismatch**: 입력/출력 형태 일치 확인
3. **Memory Error**: 배치 크기 줄이기 또는 메모리 확보

### 지원

문제가 발생하면 다음을 확인하세요:
- Python 및 패키지 버전
- 모델 구조의 복잡성
- 입력 데이터 형태


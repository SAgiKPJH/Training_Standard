# ONNX Model Sample Project

이 프로젝트는 ONNX Runtime을 활용한 모델 추론과 다양한 딥러닝 프레임워크 간 모델 변환을 위한 샘플 코드를 제공합니다.

## 프로젝트 구조

```
Onnx_Model_Sample/
├── Origin_Model/           # 원본 PyTorch 모델 (참고용)
│   ├── data/
│   │   ├── model.pth
│   │   ├── label.txt
│   │   └── inference.json
│   ├── model_handler.py
│   └── model.json
├── Onnx_Model/            # ONNX Runtime 추론 모델
│   ├── data/
│   │   ├── model.onnx     # 변환된 ONNX 모델
│   │   ├── label.txt
│   │   └── inference.json
│   ├── model_handler.py   # ONNX Runtime 핸들러
│   ├── model.json
│   └── requirements.txt
└── Convert_Onnx/          # 모델 변환 도구
    ├── pytorch_to_onnx.py
    ├── tensorflow_to_onnx.py
    ├── onnx_to_pytorch.py
    ├── onnx_to_tensorflow.py
    ├── requirements.txt
    └── README.md
```

## 주요 기능

### 1. ONNX Runtime 추론 (`Onnx_Model/`)
- **고성능**: ONNX Runtime의 최적화된 추론 엔진
- **크로스 플랫폼**: Windows, Linux, macOS 지원
- **GPU 가속**: CUDA 및 CPU 실행 제공자 지원
- **메모리 효율**: PyTorch 대비 낮은 메모리 사용량

### 2. 모델 변환 도구 (`Convert_Onnx/`)
- **PyTorch ↔ ONNX**: 양방향 변환 지원
- **TensorFlow ↔ ONNX**: 양방향 변환 지원
- **자동 검증**: 변환 후 출력 정확도 검증
- **유연한 설정**: 다양한 입력 형태 및 옵션 지원

## 빠른 시작

### 1. 의존성 설치

```bash
# ONNX 모델 실행
cd Onnx_Model
pip install -r requirements.txt

# 모델 변환 도구
cd ../Convert_Onnx
pip install -r requirements.txt
```

### 2. PyTorch 모델을 ONNX로 변환

```bash
cd Convert_Onnx
python pytorch_to_onnx.py --model_path ../Origin_Model/data/model.pth --onnx_path ../Onnx_Model/data/model.onnx --verify
```

### 3. ONNX 모델로 추론

```bash
cd Onnx_Model
python model_handler.py
```

## 성능 비교

| 프레임워크 | 메모리 사용량 | 추론 속도 | 배포 용이성 |
|-----------|-------------|----------|------------|
| PyTorch   | 높음         | 보통      | 보통        |
| TensorFlow| 높음         | 보통      | 보통        |
| **ONNX**  | **낮음**    | **빠름**  | **높음**   |

## 사용 사례

1. **모바일 배포**: ONNX Runtime Mobile으로 모바일 앱에 통합
2. **웹 서비스**: ONNX.js로 브라우저에서 실행
3. **엣지 디바이스**: 경량화된 ONNX Runtime으로 IoT 디바이스 배포
4. **크로스 플랫폼**: 다양한 환경에서 동일한 모델 실행

## 주의사항

1. **모델 호환성**: 일부 복잡한 레이어는 변환되지 않을 수 있음
2. **버전 관리**: ONNX opset 버전과 런타임 버전 호환성 확인
3. **메모리**: 대용량 모델 변환 시 충분한 메모리 확보

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.

## 지원

문제가 발생하면 다음을 확인하세요:
- [ONNX Runtime 공식 문서](https://onnxruntime.ai/)
- [ONNX 공식 문서](https://onnx.ai/)
- [GitHub Issues](https://github.com/your-repo/issues)


# TensorFlow 모델 네트워크/가중치 분리 도구

이 도구는 TensorFlow 모델 파일(.h5)에서 네트워크 구조와 가중치를 분리하는 도구입니다.

## 기능

- TensorFlow 모델 파일(.h5)에서 네트워크 구조와 가중치를 분리
- 분리된 구조와 가중치를 각각 model.h5, weight.h5로 저장
- 아키텍처 정보를 JSON 형태로도 저장
- 모델 설정을 JSON으로 저장
- 분리 후 로드 테스트 기능

## 사용법

### 기본 사용법
```bash
python model_weight_separator.py --input 모델파일.h5 --output 출력폴더
```

### 테스트 포함 실행
```bash
python model_weight_separator.py --input 모델파일.h5 --output 출력폴더 --test
```

### 예시
```bash
# Sample Model에서 model1.h5 분리
python model_weight_separator.py -i ../Sample\ Model/model1.h5 -o ./output_model1

# 테스트 포함 실행
python model_weight_separator.py -i ../Sample\ Model/model1.h5 -o ./output_model1 --test
```

## 출력 파일

분리 실행 후 다음 파일들이 생성됩니다:

1. **model.h5**: 모델 구조만 포함 (가중치 없음)
   - 네트워크 아키텍처
   - 레이어 연결 정보

2. **weight.h5**: 가중치 정보만 포함
   - 모든 레이어의 가중치
   - 편향(bias) 정보

3. **architecture_info.json**: 아키텍처 정보 (JSON 형태)
   - 입력/출력 형태
   - 레이어 이름 및 타입
   - 파라미터 수 통계
   - 가중치 형태 정보

4. **model_config.json**: 모델 설정 정보 (JSON 형태)
   - Keras 모델 설정
   - 레이어별 상세 설정

## 분리된 모델 로드 방법

```python
import tensorflow as tf

# 모델 구조 로드
model = tf.keras.models.load_model('model.h5')

# 가중치 로드 및 설정
model.load_weights('weight.h5')

# 모델 정보 확인
print(f"총 파라미터 수: {model.count_params():,}")
model.summary()
```

## 지원하는 모델 형태

- Keras Sequential 모델
- Keras Functional API 모델
- 서브클래싱 모델 (일부 제한 있음)
- 사전 훈련된 모델 (ImageNet 등)

## 파라미터 정보

분리 과정에서 다음 정보들이 제공됩니다:

- **총 파라미터 수**: 모델의 전체 파라미터 개수
- **훈련 가능한 파라미터 수**: 학습 가능한 파라미터 개수
- **훈련 불가능한 파라미터 수**: 고정된 파라미터 개수
- **레이어 수**: 모델의 레이어 개수
- **가중치 배열 수**: 가중치 텐서의 개수

## 로깅

- 클라우드 환경에서 사용하기 위해 logging 모듈 사용
- JOB_LOGGER 전역 변수 지원
- 모든 진행 상황을 로그로 출력

## 주의사항

- 분리된 모델 구조와 가중치는 서로 호환되어야 합니다
- 커스텀 레이어나 함수를 사용한 모델의 경우 완전한 재현이 어려울 수 있습니다
- 모델 로드 시 필요한 의존성이 설치되어 있어야 합니다

## 활용 사례

- 모델 구조 분석
- 가중치 전이 학습
- 모델 압축 및 최적화
- 모델 버전 관리 
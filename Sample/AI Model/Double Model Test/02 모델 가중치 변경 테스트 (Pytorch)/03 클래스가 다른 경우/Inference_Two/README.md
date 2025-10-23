# ModelHandler Pre-loading 테스트 환경

이 디렉토리는 **MasterData 기반 동적 Model Handler (Pre-loading 방식)**을 테스트할 수 있는 완전한 환경을 제공합니다.

## 🏗️ 프로젝트 구조

```
model_handler/
├── model_handler.py          # 메인 ModelHandler (Pre-loading 방식)
├── test_model_handler.py     # 테스트 스크립트
├── requirements.txt          # 필요한 패키지들
├── README.md                 # 이 파일
├── 1/                        # 폴더 1 (분류 모델)
│   └── model_handler.py      # 더미 분류 모델 핸들러
└── 2/                        # 폴더 2 (객체 검출 모델)
    └── model_handler.py      # 더미 객체 검출 모델 핸들러
```

## 🚀 빠른 시작

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 테스트 실행
```bash
cd model_handler
python test_model_handler.py
```

## 📊 테스트 내용

### 1. **Pre-loading 테스트**
- 폴더 1, 2의 model_handler를 미리 로드
- 각 폴더의 working directory에서 올바른 초기화 확인
- 상대 경로 파일 로딩 시뮬레이션

### 2. **MasterData 기반 선택 테스트**
다양한 master_data 조합으로 올바른 모델 폴더 선택 확인:

| 시나리오 | MasterData | 예상 폴더 |
|---------|------------|-----------|
| SETUP_A | `{"setup_id": "SETUP_A"}` | 폴더 1 |
| SETUP_B | `{"setup_id": "SETUP_B"}` | 폴더 2 |
| STEP_HIGH | `{"step_id": "STEP_HIGH_PRIORITY"}` | 폴더 2 |
| LINE2 | `{"line_id": "LINE2"}` | 폴더 2 |
| 기본값 | `{"setup_id": "UNKNOWN"}` | 폴더 1 |

### 3. **성능 벤치마크**
- 초기화 시간 측정
- 추론 속도 측정 (20회 반복)
- 평균/최소/최대 시간 분석

## 📈 예상 결과

### 성능 지표
- **초기화 시간**: ~1-5초 (실제 PyTorch 모델 로딩)
- **추론 시간**: ~10-50ms (실제 AI 추론)
- **폴더 1**: 실제 PyTorch 추론 시간
- **폴더 2**: 실제 PyTorch 추론 시간

### 로그 출력 예시
```
🚀 Pre-loading Dynamic ModelHandler - Loading all sub-handlers...
📥 Pre-loading handler from folder: 1
✅ Handler from folder 1 loaded in 0.1234s
📥 Pre-loading handler from folder: 2
✅ Handler from folder 2 loaded in 0.0987s
✅ Pre-loading completed! Loaded handlers: ['1', '2']

🎯 시나리오 테스트
1. SETUP_A → 폴더 1
   ✅ 예상: 1, 실제: 1, 시간: 0.0012s

⚡ 성능 벤치마크
📊 성능 결과: 평균=0.0015s, 최소=0.0011s, 최대=0.0023s
```

## 🔧 모델 특성

### 폴더 1 (실제 PyTorch 모델)
- **특화**: 실제 AI 모델 (외부 코드)
- **입력**: pickle로 직렬화된 299x299x3 numpy array
- **출력**: pickle로 직렬화된 결과 (result_code, score)
- **대상**: SETUP_A, SETUP_001
- **파일**: `./model0.pth`, `./label.txt` (실제 모델 파일)

### 폴더 2 (실제 PyTorch 모델)
- **특화**: 실제 AI 모델 (외부 코드)
- **입력**: pickle로 직렬화된 299x299x3 numpy array  
- **출력**: pickle로 직렬화된 결과 (result_code, score)
- **대상**: SETUP_B, SETUP_002, STEP_HIGH
- **파일**: `./model0.pth`, `./label.txt` (실제 모델 파일)

## 🎯 주요 기능

### 1. **Pre-loading 방식**
```python
# 초기화 시 모든 핸들러 로드
handler = ModelHandler(data, context)
# → 1, 2 폴더의 모든 model_handler 미리 로드

# 매번 빠른 추론
result, output_context = handler(data, context)
# → context의 master_data에 따라 적절한 핸들러 선택
```

### 2. **Dynamic Folder Selection**
```python
def _determine_model_folder(self, master_data):
    setup_id = master_data.get("setup_id", "")
    
    if setup_id in ["SETUP_A", "SETUP_001"]:
        return "1"  # 분류 모델
    elif setup_id in ["SETUP_B", "SETUP_002"]:
        return "2"  # 객체 검출 모델
    # ... 기타 로직
```

### 3. **Working Directory 보장**
```python
# 각 폴더의 working directory에서 초기화
with self._working_directory(model_folder_path):
    handler_instance = module.ModelHandler(data, context)
    # → 상대 경로 파일들 올바르게 로딩
```

## 📝 파일 설명

- **`model_handler.py`**: 메인 디스패처, Pre-loading 로직
- **`test_model_handler.py`**: 종합 테스트 스크립트
- **`1/model_handler.py`**: 실제 PyTorch 모델 (외부 코드, 수정 금지)
- **`2/model_handler.py`**: 실제 PyTorch 모델 (외부 코드, 수정 금지)

## 🎛️ 설정 변경

### 폴더 추가
```python
# _preload_all_handlers에서
folders_to_load = ["1", "2", "3"]  # 폴더 3 추가
```

### 선택 로직 변경
```python
# _determine_model_folder에서 비즈니스 로직 수정
if line_id == "SPECIAL_LINE":
    return "3"
```

## 🚀 실제 환경 적용

이 테스트 환경은 실제 Inference Worker 환경을 시뮬레이션합니다:

1. **Inference Worker 시작 시**: `ModelHandler` 한 번 초기화
2. **각 추론 요청 시**: `handler(data, context)` 호출
3. **MasterData에 따라**: 적절한 모델 자동 선택
4. **매우 빠른 응답**: Pre-loading으로 지연 시간 최소화

## 📊 로그 파일

테스트 실행 후 `test_model_handler.log` 파일에서 상세한 로그를 확인할 수 있습니다.

---

**🎉 이제 완전한 Pre-loading 기반 Dynamic ModelHandler를 테스트해보세요!** 
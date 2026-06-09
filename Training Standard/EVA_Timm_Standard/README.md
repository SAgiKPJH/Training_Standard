# EVA_Timm_Standard

timm `eva_large_patch14_196.in22k_ft_in22k_in1k` 모델의 **간단 학습 + 추론 테스트** 독립 폴더.

- 모델: EVA-Large (patch14, 입력 **196×196**, 파라미터 약 **303M**)
- 정규화: CLIP mean/std (모델 pretrained_cfg 값)
- **BGR 유지**: `cv2.imread` 의 BGR 를 RGB 로 변환하지 않고 학습/추론 모두 동일하게 사용
- 사전학습 가중치는 timm → **HuggingFace Hub** 에서 자동 다운로드(약 1.2GB)

## 파일 구성

| 파일 | 설명 |
|------|------|
| `eva_common.py` | 모델 생성·BGR 전처리·데이터셋·체크포인트 공통 모듈 (학습/추론 일관성 보장) |
| `train_eva.py` | 간단 파인튜닝 → `output/model.pth` 저장 |
| `inference_eva.py` | 저장된 체크포인트로 단일 이미지 추론 |
| `download_pretrained.py` | 사전학습 가중치 미리 다운로드(오프라인 대비) |
| `requirements.txt` | 의존성 |

---

## 1. .venv 구성

Windows PowerShell:

```powershell
cd "Training Standard\EVA_Timm_Standard"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux / macOS:

```bash
cd "Training Standard/EVA_Timm_Standard"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 2. 의존성 설치 (requirements.txt)

`torch` 는 플랫폼에 맞게 먼저 설치하는 것을 권장합니다.

```powershell
# (A) CPU 전용
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# (B) CUDA 12.1 예시 (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 나머지 의존성
pip install -r requirements.txt
```

> `requirements.txt` 만으로도 설치되지만, GPU 를 쓰려면 위 (B) 처럼 CUDA 빌드 torch 를 먼저 설치하세요.

## 3. 사전학습 모델 다운로드 (추론 전 필요 파일)

`train_eva.py` 는 `pretrained=True` 로 **최초 1회 자동 다운로드**합니다.
오프라인/서버 배포 전에 미리 받아두려면:

```powershell
python download_pretrained.py
```

다운로드되는 것:

- **EVA-Large 사전학습 가중치** (`model.safetensors`, 약 1.2GB)
- 저장 위치(캐시):
  - Windows: `C:\Users\<user>\.cache\huggingface\hub`
  - Linux/macOS: `~/.cache/huggingface/hub`
  - 환경변수 `HF_HOME` 로 변경 가능

> **추론 시 필요한 파일**
> - 학습으로 생성된 체크포인트 `output/model.pth` (가중치 + 클래스 목록 + 입력크기 + 정규화값 포함)
> - 추론은 `model.pth` 만 있으면 동작하며, **HuggingFace 사전학습 가중치 재다운로드가 필요 없습니다** (`pretrained=False` 로 복원).

## 4. 학습 테스트

```powershell
python train_eva.py
```

- 데이터셋: `..\..\create_dataset\dataset` (클래스별 하위폴더 구조) 사용
- 결과: `output/model.pth` 저장
- 빠른 테스트를 위해 `train_eva.py` 상단 상수로 조절:
  - `EPOCH`, `BATCH_SIZE`, `LR`, `MAX_ITER`(epoch 당 최대 iteration, `None` 이면 전체)
  - GPU 가 있으면 자동으로 CUDA 사용

## 5. 추론 테스트

```powershell
# 데이터셋 첫 이미지로 자동 추론
python inference_eva.py

# 특정 이미지
python inference_eva.py "경로\image.png"

# 이미지 + 체크포인트 지정
python inference_eva.py "경로\image.png" "경로\model.pth"
```

출력 예:

```
Image : ...\0_normal_0000.png
Pred  : 0_normal  (97.30%)
All   : 0_normal=97.30%, 1_dot=1.20%, 2_line=0.80%, 3_distortion=0.70%
```

---

## 참고

- 입력 크기는 **196 고정**(patch14 구조 요구). 다른 크기로 바꾸려면 `eva_common.INPUT_SIZE` 수정 후 동작 확인 필요.
- 정규화 mean/std 는 RGB 기준값이지만 프로젝트 **BGR 유지 규칙**에 따라 채널 변환 없이 그대로 적용합니다. 학습·추론이 동일하므로 일관성은 유지됩니다.
- 303M 파라미터 모델이라 CPU 학습은 느립니다. 테스트 용도로는 `MAX_ITER` 를 작게 두세요.

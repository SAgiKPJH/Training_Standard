# EVA_Timm_Standard

EVA-Large (patch14, 196×196) 파인튜닝 표준. BGR 유지, CLIP 정규화.

## 환경 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # GPU
pip install -r requirements.txt
```

## 사전학습 가중치 다운로드 (오프라인 환경 전 1회)

```powershell
python download_pretrained.py
# → data/pretrained/model.safetensors 저장 (약 1.2GB)
```

## 학습

```powershell
python train_eva.py
# → output/model.pth 저장
```

## 추론

```powershell
python inference_eva.py
python inference_eva.py <이미지경로>
python inference_eva.py <이미지경로> <체크포인트경로>
```

## 구조

```
EVA_Timm_Standard/
├── eva_common.py          # 공통 설정·전처리·데이터셋·체크포인트
├── train_eva.py           # 학습
├── inference_eva.py       # 추론
├── download_pretrained.py # 사전학습 가중치 다운로드
├── data/pretrained/       # model.safetensors (download_pretrained.py 실행 후 생성)
└── output/model.pth       # 학습 결과 체크포인트
```

## 참고

- `data/pretrained/model.safetensors` 없으면 HuggingFace Hub 에서 자동 다운로드
- 추론 시 `output/model.pth` 만 있으면 동작 (HuggingFace 재다운로드 불필요)
- CPU 학습은 느림 — `train_eva.py` 의 `MAX_ITER` 를 작게 설정하여 테스트

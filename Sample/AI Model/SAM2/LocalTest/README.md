## 설치 방법

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 다운로드

#### SAM2 모델
```bash
python download_sam2_model.py
```

#### SAM 모델 (기존 버전)
```bash
python download_sam_model.py
```
- 다음 모델 중 하나를 사용
  - `sam2.1_hiera_large.pt`
  - `sam2.1_hiera_base.pt`
  - `sam2.1_hiera_tiny.pt`
  - `sam2.1_hiera_base_plus.pt`

## 사용법

```bash
python model_handler.py
```

## 참조
- https://github.com/facebookresearch/sam2
- https://csm-kr.tistory.com/218
"""
EVA_Timm_Standard 공통 모듈
===========================
timm `eva_large_patch14_196.in22k_ft_in22k_in1k` 모델의 학습/추론에서
공유하는 설정·전처리·데이터셋·체크포인트 유틸.

핵심 규칙
- BGR 유지: cv2.imread 가 반환하는 BGR 를 RGB 로 변환하지 않고 그대로 사용 (학습/추론 동일)
- 전처리 일관성: 학습과 추론이 동일한 resize/normalize 를 사용하도록 한 곳에 정의
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ── 모델 설정 ─────────────────────────────────────────────
MODEL_NAME = "eva_large_patch14_196.in22k_ft_in22k_in1k"
INPUT_SIZE = 196  # patch14 / 196x196 고정
# 모델 pretrained_cfg 의 정규화 값 (CLIP mean/std). RGB 순서 기준값이지만
# BGR 유지 규칙에 따라 채널 변환 없이 그대로 적용한다 (학습/추론 동일하므로 일관).
NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
NORM_STD = (0.26862954, 0.26130258, 0.27577711)

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _load_local_weights(model, path):
    """로컬 가중치 파일(.safetensors / .pth / .bin)을 모델에 로드. head 불일치 무시."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys (head — 정상): {missing}")


def build_model(num_classes, pretrained, device="cpu", pretrained_path=None):
    """timm EVA 모델 생성.
    pretrained_path 파일이 있으면 로컬 우선 로드, 없으면 HuggingFace Hub 다운로드."""
    import timm
    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"Loading local pretrained: {pretrained_path}")
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
        _load_local_weights(model, pretrained_path)
    else:
        print(f"Loading pretrained model: {MODEL_NAME} ...")
        model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)
    return model.to(device)


def preprocess_bgr(image_bgr):
    """BGR(uint8, HxWx3) 이미지를 모델 입력 텐서(3xHxW, float32)로 변환. RGB 변환 없음."""
    img = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(NORM_MEAN, np.float32)) / np.array(NORM_STD, np.float32)
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def list_classes(dataset_path):
    """dataset_path 하위 폴더명을 정렬하여 클래스 목록으로 반환."""
    classes = [d for d in os.listdir(dataset_path)
               if os.path.isdir(os.path.join(dataset_path, d))]
    classes.sort()
    if not classes:
        raise ValueError(f"클래스 폴더가 없습니다: {dataset_path}")
    return classes


def build_samples(dataset_path, classes):
    """(이미지경로, 라벨인덱스) 리스트 생성."""
    samples = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_path, cls)
        for f in os.listdir(cls_dir):
            if f.lower().endswith(IMG_EXT):
                samples.append((os.path.join(cls_dir, f), idx))
    if not samples:
        raise ValueError(f"이미지가 없습니다: {dataset_path}")
    return samples


class BGRClassificationDataset(Dataset):
    """cv2 로 BGR 이미지를 읽어 전처리하는 분류 데이터셋."""

    def __init__(self, samples):
        self.__samples = samples

    def __len__(self):
        return len(self.__samples)

    def __getitem__(self, index):
        path, label = self.__samples[index]
        buf = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR, 한국어 경로 대응
        if image is None:
            raise RuntimeError(f"이미지 로드 실패: {path}")
        return preprocess_bgr(image), label


def save_checkpoint(path, model, classes):
    """추론에 필요한 메타정보를 함께 저장."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "model_name": MODEL_NAME,
        "input_size": INPUT_SIZE,
        "norm_mean": NORM_MEAN,
        "norm_std": NORM_STD,
        "classes": classes,
        "num_classes": len(classes),
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path, device="cpu"):
    """체크포인트로부터 모델과 클래스 목록 복원 (pretrained 다운로드 불필요)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["num_classes"], pretrained=False, device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]

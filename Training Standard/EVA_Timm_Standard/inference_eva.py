"""
EVA 추론 테스트
===============
저장된 체크포인트(model.pth)로 단일 이미지를 추론한다.
학습과 동일한 BGR + 전처리(eva_common)를 사용한다.

실행:
    python inference_eva.py                 # output/model.pth + 데이터셋 첫 이미지 자동 사용
    python inference_eva.py <이미지경로>     # 특정 이미지 추론
    python inference_eva.py <이미지경로> <체크포인트경로>
"""
import os
import sys
import logging
import cv2
import torch
import torch.nn.functional as F

import eva_common as ec

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(_HERE, "output", "model.pth")
DATASET_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", "create_dataset", "dataset"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _first_dataset_image():
    classes = ec.list_classes(DATASET_PATH)
    samples = ec.build_samples(DATASET_PATH, classes)
    return samples[0][0]


def predict(image_path, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"체크포인트가 없습니다. 먼저 train_eva.py 실행: {ckpt_path}")

    model, classes = ec.load_checkpoint(ckpt_path, device=device)

    image = cv2.imread(image_path)  # BGR, 변환 없음
    if image is None:
        raise RuntimeError(f"이미지 로드 실패: {image_path}")
    tensor = ec.preprocess_bgr(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    idx = int(torch.argmax(probs))

    logger.info(f"Image : {image_path}")
    logger.info(f"Pred  : {classes[idx]}  ({probs[idx].item()*100:.2f}%)")
    logger.info("All   : " + ", ".join(f"{c}={p.item()*100:.2f}%" for c, p in zip(classes, probs)))
    return classes[idx], probs.cpu().tolist()


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else _first_dataset_image()
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CKPT
    predict(image_path, ckpt_path)


if __name__ == "__main__":
    main()

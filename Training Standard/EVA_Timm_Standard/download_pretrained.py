"""
사전학습 가중치 다운로드
========================
eva_large_patch14_196.in22k_ft_in22k_in1k 의 pretrained 가중치를
data/pretrained/ 폴더에 저장한다. (약 1.2GB)

오프라인 환경에서 학습 전에 한 번 실행.
이후 train_eva.py 는 data/pretrained/model.safetensors 를 자동으로 사용한다.

실행:
    python download_pretrained.py
"""
import os
import logging
import eva_common as ec

_HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(_HERE, "data", "pretrained")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger.info(f"Downloading '{ec.MODEL_NAME}' -> {SAVE_DIR}")

    from huggingface_hub import hf_hub_download
    repo_id = f"timm/{ec.MODEL_NAME}"
    path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir=SAVE_DIR)
    logger.info(f"Done: {path}")
    logger.info("이후 train_eva.py 가 이 파일을 자동으로 사용합니다.")


if __name__ == "__main__":
    main()

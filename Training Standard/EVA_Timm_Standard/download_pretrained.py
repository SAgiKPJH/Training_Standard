"""
사전학습 가중치 다운로드
========================
eva_large_patch14_196.in22k_ft_in22k_in1k 의 pretrained 가중치를
HuggingFace 캐시에 미리 받아둔다. (약 1.2GB)

오프라인 환경/추론 서버에 배포하기 전에 한 번 실행하면,
이후 train_eva.py(pretrained=True) 가 네트워크 없이 동작한다.

캐시 위치(기본):
    Windows : C:\\Users\\<user>\\.cache\\huggingface\\hub
    Linux   : ~/.cache/huggingface/hub
환경변수 HF_HOME 로 변경 가능.

실행:
    python download_pretrained.py
"""
import os
import logging
import eva_common as ec

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    cache = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    logger.info(f"Downloading '{ec.MODEL_NAME}' weights -> {cache}")
    # num_classes 는 헤드 크기일 뿐, 백본 가중치 다운로드에는 영향 없음
    ec.build_model(num_classes=1000, pretrained=True, device="cpu")
    logger.info("Done. 가중치가 HuggingFace 캐시에 저장되었습니다.")


if __name__ == "__main__":
    main()

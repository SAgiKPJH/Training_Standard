import os
import numpy as np
from .inference import OcrInference

# PaddleOCR 2.7.x는 det=False여도 초기화 시 det/cls 모델을 로드.
# data/det_model, data/cls_model 에 사전 다운로드된 모델이 있으면 사용하고,
# 없으면 PaddleOCR 기본 캐시(~/.paddleocr)에서 자동 다운로드.
_DET_SUBDIR = 'data/det_model'
_CLS_SUBDIR = 'data/cls_model'


class PaddleOcrInference(OcrInference):
    """
    PaddleOCR 텍스트 인식 추론 (det=False, rec=True).
    입력: BGR 크롭 이미지 (text region already cropped).
    출력: [{'text': str, 'score': float}, ...]
    """

    def __init__(self, base_dir: str,
                 rec_model_dir: str, rec_char_dict_path: str,
                 max_text_length: int, text_thresh: float,
                 use_gpu: bool = False, gpu_id: int = 0):
        from paddleocr import PaddleOCR

        def _local(subdir):
            p = os.path.join(base_dir, subdir)
            return p if os.path.isdir(p) and os.listdir(p) else None

        self._text_thresh = text_thresh
        self._ocr = PaddleOCR(
            det_model_dir=_local(_DET_SUBDIR),
            cls_model_dir=_local(_CLS_SUBDIR),
            rec_model_dir=rec_model_dir,
            rec_char_dict_path=rec_char_dict_path,
            max_text_length=max_text_length,
            drop_score=0,
            use_gpu=use_gpu,
            gpu_mem=500 if use_gpu else 0,
            gpu_id=gpu_id,
            det=False,
            rec=True,
            lang='en',
            show_log=False,
        )

    def _run(self, image: np.ndarray) -> list:
        result = self._ocr.ocr(image, det=False, rec=True, cls=False)
        if not result or not result[0]:
            return []
        output = []
        for item in result[0]:
            text, score = item
            if float(score) >= self._text_thresh:
                output.append({'text': text, 'score': float(score)})
        return output

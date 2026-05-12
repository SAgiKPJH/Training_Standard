import numpy as np
from .inference import OcrInference


class PaddleOcrDetInference(OcrInference):
    """
    PaddleOCR 텍스트 감지+인식 추론 (det=True, rec=True).
    입력: BGR 전체 이미지.
    출력: [{'text': str, 'score': float, 'box': [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]
    """

    def __init__(self, det_model_dir: str, cls_model_dir: str,
                 rec_model_dir: str, rec_char_dict_path: str,
                 use_angle_cls: bool,
                 det_db_box_thresh: float, det_db_unclip_ratio: float,
                 max_text_length: int, text_thresh: float,
                 use_gpu: bool = False, gpu_id: int = 0):
        from paddleocr import PaddleOCR
        self._text_thresh = text_thresh
        self._ocr = PaddleOCR(
            det=True,
            det_model_dir=det_model_dir,
            cls_model_dir=cls_model_dir,
            rec=True,
            rec_model_dir=rec_model_dir,
            rec_char_dict_path=rec_char_dict_path,
            use_angle_cls=use_angle_cls,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
            max_text_length=max_text_length,
            drop_score=0,
            use_gpu=use_gpu,
            gpu_mem=500 if use_gpu else 0,
            gpu_id=gpu_id,
            lang='en',
            show_log=False,
        )

    def _run(self, image: np.ndarray) -> list:
        result = self._ocr.ocr(image, det=True, rec=True, cls=False)
        if not result or not result[0]:
            return []
        output = []
        for item in result[0]:
            box, (text, score) = item
            if float(score) >= self._text_thresh:
                output.append({'text': text, 'score': float(score), 'box': box})
        return output

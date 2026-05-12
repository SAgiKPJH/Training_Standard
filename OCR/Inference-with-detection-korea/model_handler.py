import json
import os
import time
import cv2
import pickle

from src import OcrInference, PaddleOcrDetInference


class ModelHandler:
    """
    OCR 감지+인식 추론 핸들러 (det=True, rec=True).

    모델 교체 방법:
        1. data/det_model/ 에 det 모델 파일 (.pdmodel, .pdiparams) 을 넣는다.
        2. data/rec_model/ 에 rec 모델 파일 (.pdmodel, .pdiparams) 을 넣는다.
        3. model.json 의 경로를 필요 시 수정한다.
        → 코드 수정 불필요.

    입력: BGR 전체 이미지 (pickle)
    출력: [{'text': str, 'score': float, 'box': [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...] (pickle)
    """

    def __init__(self, data, context):
        self._base_dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(self._base_dir, "model.json"), 'r', encoding='utf-8') as f:
            config = json.load(f)

        def _path(key):
            v = config.get(key)
            return os.path.join(self._base_dir, v) if v else None

        det_model_dir      = _path('det_model_dir')
        cls_model_dir      = _path('cls_model_dir')
        rec_model_dir      = _path('rec_model_dir')
        rec_char_dict_path = _path('rec_char_dict_path')
        use_angle_cls      = bool(config.get('use_angle_cls', False))
        det_db_box_thresh  = float(config.get('det_db_box_thresh', 0.3))
        det_db_unclip_ratio = float(config.get('det_db_unclip_ratio', 2.3))
        max_text_length    = int(config.get('max_text_length', 50))
        text_thresh        = float(config.get('text_thresh', 0.9))
        use_gpu            = bool(config.get('use_gpu', False))
        gpu_id             = int(config.get('gpu_id', 0))
        self._test_image_path = os.path.join(self._base_dir, config.get('test_image', 'data/test_image.png'))

        self._inference: OcrInference = PaddleOcrDetInference(
            det_model_dir, cls_model_dir, rec_model_dir, rec_char_dict_path,
            use_angle_cls, det_db_box_thresh, det_db_unclip_ratio,
            max_text_length, text_thresh,
            use_gpu=use_gpu, gpu_id=gpu_id,
        )

    def __call__(self, data, context):
        image = pickle.loads(data)
        output = self._inference.infer(image)
        return pickle.dumps(output), context

    def test(self):
        if not os.path.exists(self._test_image_path):
            raise FileNotFoundError(f"테스트 이미지를 찾을 수 없습니다: {self._test_image_path}")

        image = cv2.imread(self._test_image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {self._test_image_path}")

        h, w = image.shape[:2]
        print(f"[test] image  : {self._test_image_path}")
        print(f"[test] size   : {w}x{h} px")

        start = time.perf_counter()
        result_bytes, _ = self(pickle.dumps(image), None)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = pickle.loads(result_bytes)
        print(f"[test] elapsed: {elapsed_ms:.1f} ms")
        print(f"[test] found  : {len(results)} region(s)")
        for item in results:
            print(f"  text={item['text']!r}  score={item['score']:.4f}  box={item['box']}")


if __name__ == "__main__":
    handler = ModelHandler(None, None)
    handler.test()

"""
EVA-Large 가중치 전용 추론 핸들러
===================================
가중치만 저장된 .pth (state_dict) 를 사용하는 경우.
메타정보가 없으므로 model.json 에 클래스·입력크기·정규화값을 직접 기입해야 한다.

모델 교체:
    1. data/ 폴더에 가중치 파일을 넣는다.
    2. model.json 의 model_file·classes 를 수정한다.
    → 코드 수정 불필요.
"""
import json
import os
import time
import cv2
import pickle
import numpy as np

from src import EVAInference


class ModelHandler:

    def __init__(self, data, context):
        self._base_dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(self._base_dir, "model.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        model_file = os.path.join(self._base_dir, config["model_file"])
        self._inference = EVAInference(model_file, config)
        self._test_image_path = os.path.join(
            self._base_dir, config.get("test_image", "data/image.png"))

    def __call__(self, data, context):
        image = pickle.loads(data)
        return pickle.dumps(self._inference.infer(image)), context

    def test(self):
        image = None
        if os.path.exists(self._test_image_path):
            buf = np.fromfile(self._test_image_path, dtype=np.uint8)
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        if image is None:
            size = self._inference.input_size
            print(f"[WARN] 테스트 이미지 없음. 랜덤 이미지 생성 ({size}x{size})")
            image = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
            image_source = "random"
        else:
            image_source = self._test_image_path

        start = time.perf_counter()
        result_bytes, _ = self(pickle.dumps(image), None)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = pickle.loads(result_bytes)
        print(f"[test] image  : {image_source}")
        print(f"[test] elapsed: {elapsed_ms:.1f} ms")
        print("[test] results:")
        for item in results:
            print(f"  {item}")


if __name__ == "__main__":
    handler = ModelHandler(None, None)
    handler.test()

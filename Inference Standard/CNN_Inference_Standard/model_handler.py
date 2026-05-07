import json
import os
import time
import cv2
import pickle

from src import Inference, PytorchInference, TensorflowInference


class ModelHandler:
    """
    CNN 분류 추론 핸들러.

    모델 교체 방법:
        1. data/ 폴더에 모델 파일을 넣는다.
        2. model.json 의 "model_file" 경로를 바꾼다.
        → 코드 수정 불필요.

    지원 포맷:
        .pth / .pt       → PyTorch  (src.PytorchInference)
        .h5 / .keras     → TF Keras (src.TensorflowInference)
        디렉토리          → TF SavedModel (src.TensorflowInference)
    """

    def __init__(self, data, context):
        self._base_dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(self._base_dir, "model.json"), 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_file      = os.path.join(self._base_dir, config['model_file'])
        device          = config.get('device', 'cpu')
        normalize_mean  = float(config.get('normalize_mean', 0.5))
        normalize_stdev = float(config.get('normalize_stdev', 0.5))
        self._test_image_path = os.path.join(self._base_dir, config.get('test_image', 'data/image.png'))

        ext = os.path.splitext(model_file)[1].lower()

        if ext in ('.pth', '.pt'):
            self._inference: Inference = PytorchInference(model_file, device, normalize_mean, normalize_stdev)
        elif ext in ('.h5', '.keras') or os.path.isdir(model_file):
            self._inference: Inference = TensorflowInference(model_file, device, normalize_mean, normalize_stdev)
        else:
            raise ValueError(
                f"지원하지 않는 모델 파일: '{model_file}'\n"
                "PyTorch: .pth / .pt\n"
                "TensorFlow: .h5 / .keras / SavedModel 디렉토리"
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

        start = time.perf_counter()
        result_bytes, _ = self(pickle.dumps(image), None)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = pickle.loads(result_bytes)
        print(f"[test] image : {self._test_image_path}")
        print(f"[test] elapsed: {elapsed_ms:.1f} ms")
        print("[test] results:")
        for item in results:
            print(f"  {item}")


if __name__ == "__main__":
    handler = ModelHandler(None, None)
    handler.test()

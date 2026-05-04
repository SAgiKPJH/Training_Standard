import os
import json
import cv2
import numpy as np
from .inference import Inference, parse_inference_info


class TensorflowInference(Inference):
    """
    TensorFlow Keras 모델 추론.
    inference_info 는 모델과 같은 위치의 _info.json 사이드카 파일에서 읽는다.
    """

    def __init__(self, model_path: str, device: str,
                 normalize_mean: float, normalize_stdev: float):
        import tensorflow as tf
        self._tf = tf
        self._device = '/GPU:0' if device.lower() in ('gpu', 'cuda') else '/CPU:0'
        self._mean = normalize_mean
        self._std = normalize_stdev

        self._model = tf.keras.models.load_model(model_path)

        ext = os.path.splitext(model_path)[1]
        info_path = (model_path.rsplit('.', 1)[0] + '_info.json' if ext
                     else os.path.join(model_path, '_info.json'))
        with open(info_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        self._input_size, self._label_info = parse_inference_info(raw)

    def _run(self, image: np.ndarray) -> tuple:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img = cv2.resize(image, (self._input_size, self._input_size))
        img = (img.astype(np.float32) / 255.0 - self._mean) / self._std
        img = np.expand_dims(img, 0)

        with self._tf.device(self._device):
            out = self._model(img, training=False)
            scores = self._tf.nn.softmax(out, axis=1).numpy()[0]

        return [float(scores[i]) for i in range(len(scores))], self._label_info

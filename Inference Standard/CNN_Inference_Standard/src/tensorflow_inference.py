import os
import json
import cv2
import numpy as np
from .inference import Inference, parse_inference_info


class TensorflowInference(Inference):
    """
    TensorFlow Keras 모델 추론.
    H5 모델의 extra_info 그룹 inference_info attribute에서 읽는다.
    (mpp.daq.object_service.upload_model 및 Local_SaveBuilder와 동일한 형식)
    """

    def __init__(self, model_path: str, device: str,
                 normalize_mean: float, normalize_stdev: float):
        import tensorflow as tf
        self._tf = tf
        self._device = '/GPU:0' if device.lower() in ('gpu', 'cuda') else '/CPU:0'
        self._mean = normalize_mean
        self._std = normalize_stdev

        self._model = tf.keras.models.load_model(model_path)

        inference_info_str = None

        if model_path.endswith('.h5') and os.path.isfile(model_path):
            import h5py
            with h5py.File(model_path, 'r') as f:
                if 'extra_info' in f and 'inference_info' in f['extra_info'].attrs:
                    inference_info_str = f['extra_info'].attrs['inference_info']
                    if isinstance(inference_info_str, bytes):
                        inference_info_str = inference_info_str.decode('utf-8')

        if inference_info_str is None:
            raise ValueError(
                f"모델에 'inference_info'가 없습니다: {model_path}\n"
                f"학습 시 inference_info가 내장된 모델이 필요합니다."
            )

        raw = json.loads(inference_info_str)
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

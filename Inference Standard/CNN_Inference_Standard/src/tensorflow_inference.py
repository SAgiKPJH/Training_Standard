import os
import json
import cv2
import numpy as np
from .inference import Inference, parse_inference_info


class TensorflowInference(Inference):
    """
    TensorFlow Keras 모델 추론.
    inference_info 는 모델에 내장된 attribute/variable에서 읽는다.
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

        # 1) tf.Variable로 내장된 경우 (SavedModel/keras 포맷)
        if hasattr(self._model, 'inference_info'):
            try:
                inference_info_str = self._model.inference_info.numpy().decode('utf-8')
            except Exception:
                inference_info_str = None

        # 2) H5 파일의 attribute로 내장된 경우
        if inference_info_str is None and model_path.endswith('.h5') and os.path.isfile(model_path):
            try:
                import h5py
                with h5py.File(model_path, 'r') as f:
                    if 'inference_info' in f.attrs:
                        inference_info_str = f.attrs['inference_info']
                        if isinstance(inference_info_str, bytes):
                            inference_info_str = inference_info_str.decode('utf-8')
                    # extra_info 그룹 (구버전 DAQ OLD 방식)도 시도
                    if inference_info_str is None and 'extra_info' in f:
                        extra = f['extra_info']
                        if 'inference_info' in extra.attrs:
                            inference_info_str = extra.attrs['inference_info']
                            if isinstance(inference_info_str, bytes):
                                inference_info_str = inference_info_str.decode('utf-8')
            except Exception:
                pass

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

import cv2
import numpy as np
from .inference import Inference, parse_inference_info


class PytorchInference(Inference):
    """
    PyTorch TorchScript 모델 추론.
    inference_info 는 .pth 파일에 _extra_files 로 내장되어 있다.
    """

    def __init__(self, model_path: str, device: str,
                 normalize_mean: float, normalize_stdev: float):
        import torch
        import torchvision.transforms as transforms
        from PIL import Image

        # "gpu" → "cuda"  (TF 표기와 통일)
        if device.lower() == 'gpu':
            device = 'cuda'

        extra_files = {'inference_info': '', 'label_info': ''}
        self._model = torch.jit.load(
            model_path, map_location=device, _extra_files=extra_files
        ).eval().to(device)
        self._device = device
        self._torch = torch

        raw = {k: (v.decode('utf-8') if isinstance(v, bytes) else v)
               for k, v in extra_files.items()}
        self._input_size, self._label_info = parse_inference_info(raw)

        self._transform = transforms.Compose([
            transforms.Lambda(lambda img: Image.fromarray(img).convert("RGB")),
            transforms.Resize((self._input_size, self._input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[normalize_mean] * 3,
                std=[normalize_stdev] * 3
            )
        ])

    def _run(self, image: np.ndarray) -> tuple:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            out = self._model(tensor)
            if isinstance(out, tuple):
                out = out[0]
            out = self._torch.nn.functional.softmax(out.to('cpu'), dim=1)[0]

        return [float(out[i]) for i in range(len(out))], self._label_info

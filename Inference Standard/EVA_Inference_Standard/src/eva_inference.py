"""
EVA-Large 분류 추론
===================
Training Standard 의 save_checkpoint() 로 저장된 .pth 체크포인트를 로드한다.
체크포인트에 model_name·classes·input_size·norm_mean/std 가 내장되어 있으므로
별도 설정 없이 모델을 완전히 복원할 수 있다.

BGR 유지: cv2.imdecode 가 반환하는 BGR 을 RGB 로 변환하지 않고 그대로 사용.
"""
import cv2
import numpy as np


class EVAInference:
    """
    모델 교체 시 model.json 의 model_file 경로만 바꾸면 된다.
    num_classes·input_size·norm_mean/std·classes 는 체크포인트에서 자동 복원.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        import torch
        import timm

        if device.lower() == "gpu":
            device = "cuda"
        self._device = device
        self._torch = torch

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self._classes    = ckpt["classes"]
        self._input_size = ckpt["input_size"]
        self._norm_mean  = ckpt["norm_mean"]
        self._norm_std   = ckpt["norm_std"]

        self._model = timm.create_model(
            ckpt["model_name"], pretrained=False, num_classes=ckpt["num_classes"])
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.to(device).eval()

    @property
    def input_size(self) -> int:
        return self._input_size

    def infer(self, image_bgr: np.ndarray) -> list:
        """BGR ndarray → [{'code': int, 'name': str, 'score': float}, ...] (score 내림차순)"""
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(image_bgr, (self._input_size, self._input_size),
                         interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(self._norm_mean, np.float32)) / np.array(self._norm_std, np.float32)
        tensor = (self._torch.from_numpy(img)
                  .permute(2, 0, 1).unsqueeze(0).contiguous().to(self._device))

        with self._torch.no_grad():
            out = self._torch.nn.functional.softmax(
                self._model(tensor).cpu(), dim=1)[0]

        results = [{"code": i, "name": self._classes[i], "score": float(out[i])}
                   for i in range(len(self._classes))]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

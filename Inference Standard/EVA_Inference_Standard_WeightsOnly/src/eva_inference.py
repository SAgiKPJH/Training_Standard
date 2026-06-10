"""
EVA-Large 가중치 전용 추론
===========================
torch.save(model.state_dict(), path) 로 저장한 가중치 파일을 로드한다.
메타정보(model_name·classes·input_size·norm_mean/std)는 model.json 에서 읽는다.

BGR 유지: cv2.imdecode 가 반환하는 BGR 을 RGB 로 변환하지 않고 그대로 사용.
"""
import cv2
import numpy as np


class EVAInference:
    """
    model.json 에 아래 항목을 반드시 기입해야 한다.
        model_name, input_size, norm_mean, norm_std, classes
    """

    def __init__(self, model_path: str, config: dict):
        import torch
        import timm

        device = config.get("device", "cpu")
        if device.lower() == "gpu":
            device = "cuda"
        self._device     = device
        self._torch      = torch
        self._classes    = config["classes"]
        self._input_size = config["input_size"]
        self._norm_mean  = config["norm_mean"]
        self._norm_std   = config["norm_std"]

        model_name  = config["model_name"]
        num_classes = len(self._classes)

        self._model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [WARN] Missing keys: {missing}")
        if unexpected:
            print(f"  [WARN] Unexpected keys: {unexpected}")
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

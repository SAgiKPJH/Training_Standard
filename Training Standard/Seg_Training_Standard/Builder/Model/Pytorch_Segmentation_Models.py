import torch
import torchvision


# torchvision.models.segmentation에서 가져올 모델 이름 매핑
_TORCHVISION_SEG_MODELS = {
    "fcn_resnet50": "fcn_resnet50",
    "fcn_resnet101": "fcn_resnet101",
    "deeplabv3_resnet50": "deeplabv3_resnet50",
    "deeplabv3_resnet101": "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large": "deeplabv3_mobilenet_v3_large",
    "lraspp_mobilenet_v3_large": "lraspp_mobilenet_v3_large",
}


def _resolve_model(network_name: str):
    if network_name not in _TORCHVISION_SEG_MODELS:
        return None
    attr_name = _TORCHVISION_SEG_MODELS[network_name]
    return getattr(torchvision.models.segmentation, attr_name, None)


class _SegmentationWrapper(torch.nn.Module):
    """torchvision segmentation 모델의 OrderedDict 출력을 텐서로 풀어주는 래퍼.
    학습/추론 코드가 일반 nn.Module처럼 다룰 수 있게 한다.
    aux_classifier 출력은 무시한다.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict) and 'out' in out:
            return out['out']
        return out


class Pytorch_Segmentation_Models:
    def __init__(self):
        self.__device = 'cpu'
        self.__model = None

    def init_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            raise Exception("GPU is not available")
        self.__device = 'cuda' if device == 'cuda' else 'cpu'
        return self

    def init_model(self, num_classes, network_name=None, pretrained_backbone=True, output_stride=16, **kwargs):
        if network_name is None:
            raise ValueError("network_name is required")

        model_fn = _resolve_model(network_name)
        if model_fn is None:
            available = [n for n in _TORCHVISION_SEG_MODELS if _resolve_model(n) is not None]
            raise ValueError(
                f"Unsupported network: '{network_name}'. "
                f"This torchvision (v{torchvision.__version__}) supports: {available}"
            )

        # torchvision >= 0.13: weights_backbone=ResNet50_Weights.IMAGENET1K_V1 형태
        # 0.12 이하: pretrained_backbone=True/False
        # 양쪽 호환을 위해 여러 패턴을 순차 시도
        model = None
        last_err = None
        try_kwargs_list = []
        if pretrained_backbone:
            try_kwargs_list.append({"num_classes": num_classes, "weights_backbone": "DEFAULT", "aux_loss": False})
            try_kwargs_list.append({"num_classes": num_classes, "pretrained_backbone": True, "aux_loss": False})
        else:
            try_kwargs_list.append({"num_classes": num_classes, "weights_backbone": None, "aux_loss": False})
            try_kwargs_list.append({"num_classes": num_classes, "pretrained_backbone": False, "aux_loss": False})
        try_kwargs_list.append({"num_classes": num_classes, "aux_loss": False})
        try_kwargs_list.append({"num_classes": num_classes})

        for try_kwargs in try_kwargs_list:
            try:
                model = model_fn(**try_kwargs)
                break
            except TypeError as e:
                last_err = e
                continue
        if model is None:
            raise RuntimeError(f"Failed to instantiate {network_name}: {last_err}")

        wrapped = _SegmentationWrapper(model)
        wrapped.to(self.__device)
        self.__model = wrapped
        return self

    def get_model(self):
        return self.__model

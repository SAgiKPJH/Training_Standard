import torch
import torchvision
from .Models.InceptionV2 import InceptionV2
from .Models.InceptionV4 import InceptionV4
from .Models.B3_mirero import B3_mirero
from .Models.B7_mirero import B7_mirero
from .Models.V2S_mirero import V2S_mirero
from .Models.V2M_mirero import V2M_mirero
from .Models.V2L_mirero import V2L_mirero

# 커스텀 모델 (torchvision에 없거나 별도 정의가 필요한 모델)
_CUSTOM_MODELS = {
    "inceptionv2": InceptionV2,
    "inceptionv4": InceptionV4,
    "B3_mirero": B3_mirero,
    "B7_mirero": B7_mirero,
    "V2S_mirero": V2S_mirero,
    "V2M_mirero": V2M_mirero,
    "V2L_mirero": V2L_mirero,
}

# torchvision.models에서 가져올 모델 이름 매핑 (key: 사용자 이름, value: torchvision attr 이름)
_TORCHVISION_MODELS = {
    # ResNet
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "resnet152": "resnet152",
    # Inception
    "inceptionv1": "googlenet",
    "inceptionv3": "inception_v3",
    # EfficientNet
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "efficientnet_b6": "efficientnet_b6",
    "efficientnet_b7": "efficientnet_b7",
    "efficientnet_v2_s": "efficientnet_v2_s",
    "efficientnet_v2_m": "efficientnet_v2_m",
    "efficientnet_v2_l": "efficientnet_v2_l",
    # VGG
    "vgg11": "vgg11",
    "vgg13": "vgg13",
    "vgg16": "vgg16",
    "vgg19": "vgg19",
    "vgg11_bn": "vgg11_bn",
    "vgg13_bn": "vgg13_bn",
    "vgg16_bn": "vgg16_bn",
    "vgg19_bn": "vgg19_bn",
    # DenseNet
    "densenet121": "densenet121",
    "densenet169": "densenet169",
    "densenet201": "densenet201",
    # MobileNet
    "mobilenet_v2": "mobilenet_v2",
    "mobilenet_v3_small": "mobilenet_v3_small",
    "mobilenet_v3_large": "mobilenet_v3_large",
    # ShuffleNet
    "shufflenet_v2_x0_5": "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0": "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5": "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0": "shufflenet_v2_x2_0",
    # MNASNet
    "mnasnet0_5": "mnasnet0_5",
    "mnasnet1_0": "mnasnet1_0",
    # SqueezeNet
    "squeezenet1_0": "squeezenet1_0",
    "squeezenet1_1": "squeezenet1_1",
    # RegNet
    "regnet_y_400mf": "regnet_y_400mf",
    "regnet_y_800mf": "regnet_y_800mf",
    "regnet_y_1_6gf": "regnet_y_1_6gf",
    "regnet_y_3_2gf": "regnet_y_3_2gf",
    "regnet_y_8gf": "regnet_y_8gf",
    "regnet_y_16gf": "regnet_y_16gf",
    "regnet_y_32gf": "regnet_y_32gf",
    "regnet_x_400mf": "regnet_x_400mf",
    "regnet_x_800mf": "regnet_x_800mf",
    "regnet_x_1_6gf": "regnet_x_1_6gf",
    "regnet_x_3_2gf": "regnet_x_3_2gf",
    "regnet_x_8gf": "regnet_x_8gf",
    "regnet_x_16gf": "regnet_x_16gf",
    "regnet_x_32gf": "regnet_x_32gf",
    # ConvNeXt
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnext_base": "convnext_base",
    "convnext_large": "convnext_large",
    # Wide ResNet
    "wide_resnet50_2": "wide_resnet50_2",
    "wide_resnet101_2": "wide_resnet101_2",
    # ResNeXt
    "resnext50_32x4d": "resnext50_32x4d",
    "resnext101_32x8d": "resnext101_32x8d",
    "resnext101_64x4d": "resnext101_64x4d",
    # Swin Transformer
    "swin_t": "swin_t",
    "swin_s": "swin_s",
    "swin_b": "swin_b",
    "swin_v2_t": "swin_v2_t",
    "swin_v2_s": "swin_v2_s",
    "swin_v2_b": "swin_v2_b",
    # Vision Transformer
    "vit_b_16": "vit_b_16",
    "vit_b_32": "vit_b_32",
    "vit_l_16": "vit_l_16",
    "vit_l_32": "vit_l_32",
    "vit_h_14": "vit_h_14",
    # MaxVit
    "maxvit_t": "maxvit_t",
    # AlexNet
    "alexnet": "alexnet",
}


def _resolve_model(network_name: str):
    """network_name에 해당하는 모델 팩토리 함수를 반환. 없으면 None."""
    if network_name in _CUSTOM_MODELS:
        return _CUSTOM_MODELS[network_name]
    if network_name in _TORCHVISION_MODELS:
        attr_name = _TORCHVISION_MODELS[network_name]
        return getattr(torchvision.models, attr_name, None)
    return None




class Pytorch_Classification_Models:
    def __init__(self):
        self.__device = 'cpu'
        self.__model = None

    def init_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            raise Exception("GPU is not available")
        self.__device = 'cuda' if device == 'cuda' else 'cpu'
        return self

    def init_model(self, num_classes, network_name=None):
        if network_name is None:
            raise ValueError("network_name is required")

        model_fn = _resolve_model(network_name)
        if model_fn is None:
            all_names = list(_CUSTOM_MODELS.keys()) + list(_TORCHVISION_MODELS.keys())
            available = [n for n in all_names if _resolve_model(n) is not None]
            raise ValueError(
                f"Unsupported network: '{network_name}'. "
                f"This torchvision (v{torchvision.__version__}) supports: {available}"
            )

        if network_name == "inceptionv1":
            model = model_fn(num_classes=num_classes, init_weights=False)
        elif network_name == "inceptionv3":
            model = model_fn(num_classes=num_classes, init_weights=False, aux_logits=False)
        else:
            model = model_fn(num_classes=num_classes)

        self.__model = model
        model.to(self.__device)
        return self

    def get_model(self):
        return self.__model

import torch
import torchvision
from .Models.InceptionV2 import InceptionV2
from .Models.InceptionV4 import InceptionV4

class Pytorch_Classification_Models:
    def __init__(self):
        self.__device = 'cpu'
        self.__model = None
        self.__network_map = {
            # ResNet
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
            "resnet101": torchvision.models.resnet101,
            "resnet152": torchvision.models.resnet152,
            # Inception
            "inceptionv1": torchvision.models.googlenet,
            "inceptionv2": InceptionV2,
            "inceptionv3": torchvision.models.inception_v3,
            "inceptionv4": InceptionV4,
            # EfficientNet
            "efficientnet_b0": torchvision.models.efficientnet_b0,
            "efficientnet_b1": torchvision.models.efficientnet_b1,
            "efficientnet_b2": torchvision.models.efficientnet_b2,
            "efficientnet_b3": torchvision.models.efficientnet_b3,
            "efficientnet_b4": torchvision.models.efficientnet_b4,
            "efficientnet_b5": torchvision.models.efficientnet_b5,
            "efficientnet_b6": torchvision.models.efficientnet_b6,
            "efficientnet_b7": torchvision.models.efficientnet_b7,
            "efficientnet_v2_s": torchvision.models.efficientnet_v2_s,
            "efficientnet_v2_m": torchvision.models.efficientnet_v2_m,
            "efficientnet_v2_l": torchvision.models.efficientnet_v2_l,
            # VGG
            "vgg11": torchvision.models.vgg11,
            "vgg13": torchvision.models.vgg13,
            "vgg16": torchvision.models.vgg16,
            "vgg19": torchvision.models.vgg19,
            "vgg11_bn": torchvision.models.vgg11_bn,
            "vgg13_bn": torchvision.models.vgg13_bn,
            "vgg16_bn": torchvision.models.vgg16_bn,
            "vgg19_bn": torchvision.models.vgg19_bn,
            # DenseNet
            "densenet121": torchvision.models.densenet121,
            "densenet169": torchvision.models.densenet169,
            "densenet201": torchvision.models.densenet201,
            # MobileNet
            "mobilenet_v2": torchvision.models.mobilenet_v2,
            "mobilenet_v3_small": torchvision.models.mobilenet_v3_small,
            "mobilenet_v3_large": torchvision.models.mobilenet_v3_large,
            # ShuffleNet
            "shufflenet_v2_x0_5": torchvision.models.shufflenet_v2_x0_5,
            "shufflenet_v2_x1_0": torchvision.models.shufflenet_v2_x1_0,
            "shufflenet_v2_x1_5": torchvision.models.shufflenet_v2_x1_5,
            "shufflenet_v2_x2_0": torchvision.models.shufflenet_v2_x2_0,
            # MNASNet
            "mnasnet0_5": torchvision.models.mnasnet0_5,
            "mnasnet1_0": torchvision.models.mnasnet1_0,
            # SqueezeNet
            "squeezenet1_0": torchvision.models.squeezenet1_0,
            "squeezenet1_1": torchvision.models.squeezenet1_1,
            # RegNet
            "regnet_y_400mf": torchvision.models.regnet_y_400mf,
            "regnet_y_800mf": torchvision.models.regnet_y_800mf,
            "regnet_y_1_6gf": torchvision.models.regnet_y_1_6gf,
            "regnet_y_3_2gf": torchvision.models.regnet_y_3_2gf,
            "regnet_y_8gf": torchvision.models.regnet_y_8gf,
            "regnet_y_16gf": torchvision.models.regnet_y_16gf,
            "regnet_y_32gf": torchvision.models.regnet_y_32gf,
            "regnet_x_400mf": torchvision.models.regnet_x_400mf,
            "regnet_x_800mf": torchvision.models.regnet_x_800mf,
            "regnet_x_1_6gf": torchvision.models.regnet_x_1_6gf,
            "regnet_x_3_2gf": torchvision.models.regnet_x_3_2gf,
            "regnet_x_8gf": torchvision.models.regnet_x_8gf,
            "regnet_x_16gf": torchvision.models.regnet_x_16gf,
            "regnet_x_32gf": torchvision.models.regnet_x_32gf,
            # ConvNeXt
            "convnext_tiny": torchvision.models.convnext_tiny,
            "convnext_small": torchvision.models.convnext_small,
            "convnext_base": torchvision.models.convnext_base,
            "convnext_large": torchvision.models.convnext_large,
            # Wide ResNet
            "wide_resnet50_2": torchvision.models.wide_resnet50_2,
            "wide_resnet101_2": torchvision.models.wide_resnet101_2,
            # ResNeXt
            "resnext50_32x4d": torchvision.models.resnext50_32x4d,
            "resnext101_32x8d": torchvision.models.resnext101_32x8d,
            "resnext101_64x4d": torchvision.models.resnext101_64x4d,
            # Swin Transformer
            "swin_t": torchvision.models.swin_t,
            "swin_s": torchvision.models.swin_s,
            "swin_b": torchvision.models.swin_b,
            "swin_v2_t": torchvision.models.swin_v2_t,
            "swin_v2_s": torchvision.models.swin_v2_s,
            "swin_v2_b": torchvision.models.swin_v2_b,
            # Vision Transformer
            "vit_b_16": torchvision.models.vit_b_16,
            "vit_b_32": torchvision.models.vit_b_32,
            "vit_l_16": torchvision.models.vit_l_16,
            "vit_l_32": torchvision.models.vit_l_32,
            "vit_h_14": torchvision.models.vit_h_14,
            # MaxVit
            "maxvit_t": torchvision.models.maxvit_t,
            # AlexNet
            "alexnet": torchvision.models.alexnet,
        }
    
    def init_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            raise Exception("GPU is not available")
        self.__device = 'cuda' if device == 'cuda' else 'cpu'
        return self
    
    def init_model(self, num_classes, network_name=None, aux_logits=False):
        if network_name is None:
            raise ValueError("network_name is required")
        if network_name not in self.__network_map:
            raise ValueError(f"Unsupported network name: {network_name}. Supported names are: {list(self.__network_map.keys())}")

        if network_name == "inceptionv1":
            model = self.__network_map[network_name](num_classes=num_classes, init_weights=False)
        elif network_name == "inceptionv3":
            model = self.__network_map[network_name](num_classes=num_classes, init_weights=False, aux_logits=aux_logits)
        else:
            model = self.__network_map[network_name](num_classes=num_classes)

        self.__model = model
        model.to(self.__device)
        return self
    
    def get_model(self):
        return self.__model
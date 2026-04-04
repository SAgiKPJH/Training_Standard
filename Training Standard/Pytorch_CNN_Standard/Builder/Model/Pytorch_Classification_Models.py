import torch
import torchvision
from .Models.InceptionV2 import InceptionV2
from .Models.InceptionV4 import InceptionV4

class Pytorch_Classification_Models:
    def __init__(self):
        self.__device = 'cpu'
        self.__model = None
        self.__network_map = {
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
            "inceptionv1": torchvision.models.googlenet,
            "inceptionv2": InceptionV2,
            "inceptionv3": torchvision.models.inception_v3,
            "inceptionv4": InceptionV4,
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
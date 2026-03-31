import torch
import torchvision

class Pytorch_Efficientnet:
    def __init__(self):
        self.__device = 'cpu'
        self.__model = None
        self.__efficientnet_map = {
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
    
    def init_model(self, num_classes, network_name='efficientnet_b0'):
        if network_name not in self.__efficientnet_map:
            raise ValueError(f"Unsupported network name: {network_name}. Supported names are: {list(self.__efficientnet_map.keys())}")

        model = self.__efficientnet_map[network_name](num_classes=num_classes)
        
        self.__model = model
        model.to(self.__device)
        return self
    
    def get_model(self):
        return self.__model
import torch
import torchvision

class DAQ_Pytorch_InceptionV3:
    def __init__(self):
        self.__device = 'cpu'
        self.__model = None
    
    def init_device(self, device):
        if device and not torch.cuda.is_available():
            raise Exception("GPU is not available")
        self.__device = 'cuda' if device else 'cpu'
        return self
    
    def init_model(self, num_classes):
        model = torchvision.models.inception_v3(num_classes=num_classes, init_weights=False)
        self.__model = model
        model.to(self.__device)
        return self
    
    def get_model(self):
        return self.__model
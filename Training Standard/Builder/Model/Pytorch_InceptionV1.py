import torch
import torchvision

class Pytorch_InceptionV1:
    def __init__(self):
        self.__device = 'cpu'
        self.__model = None
    
    def init_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            raise Exception("GPU is not available")
        self.__device = 'cuda' if device == 'cuda' else 'cpu'
        return self
    
    def init_model(self, num_classes):
        model = torchvision.models.googlenet(num_classes=num_classes, init_weights=False)
        self.__model = model
        model.to(self.__device)
        return self
    
    def get_model(self):
        return self.__model

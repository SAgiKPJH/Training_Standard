"""
Inception V1 (GoogLeNet) Model
================================

Architecture:
- Also known as GoogLeNet
- Introduced the Inception module concept
- Uses 1x1, 3x3, 5x5 convolutions and 3x3 max pooling in parallel
- 22 layers deep
- Uses global average pooling instead of fully connected layers
- Auxiliary classifiers for training (removed during inference)

Key Features:
- Efficient computation through 1x1 convolutions for dimensionality reduction
- Multi-scale feature extraction through parallel convolutions
- Reduced parameters compared to VGG

Input:
- Shape: (batch_size, 3, 224, 224)
- Color: RGB images
- Normalization: Typically normalized to [0, 1] or [-1, 1]

Output:
- Shape: (batch_size, num_classes)
- Logits for each class (before softmax)

Example Usage:
    model = Pytorch_InceptionV1()
    model.init_device('cuda').init_model(num_classes=1000)

    # Forward pass
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model.get_model()(input_tensor)
    # output shape: (1, 1000)
"""

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
"""
Inception V3 Model
==================

Architecture:
- Improved version of Inception V1
- Factorizes 5x5 convolutions into two 3x3 convolutions
- Factorizes nxn convolutions into 1xn and nx1 convolutions
- Uses efficient grid size reduction
- Label smoothing regularization
- Batch normalization in auxiliary classifiers

Key Features:
- More efficient computation through factorized convolutions
- Better gradient flow through auxiliary classifiers
- Improved regularization techniques
- Higher accuracy with similar computational cost

Input:
- Shape: (batch_size, 3, 299, 299)
- Color: RGB images (NOTE: Larger input size than V1)
- Normalization: Typically normalized to [0, 1] or [-1, 1]

Output:
- Shape: (batch_size, num_classes)
- Logits for each class (before softmax)

Note:
- During training, returns tuple (main_output, aux_output) if aux_logits=True
- During evaluation, returns only main_output

Example Usage:
    model = Pytorch_InceptionV3()
    model.init_device('cuda').init_model(num_classes=1000)

    # Forward pass
    input_tensor = torch.randn(1, 3, 299, 299)
    output = model.get_model()(input_tensor)
    # output shape: (1, 1000)
"""

import torch
import torchvision

class Pytorch_InceptionV3:
    def __init__(self):
        self.__device = 'cpu'
        self.__model = None
    
    def init_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            raise Exception("GPU is not available")
        self.__device = 'cuda' if device == 'cuda' else 'cpu'
        return self
    
    def init_model(self, num_classes):
        model = torchvision.models.inception_v3(num_classes=num_classes, init_weights=False)
        self.__model = model
        model.to(self.__device)
        return self
    
    def get_model(self):
        return self.__model
"""
Inception V2 Model (BN-Inception)
==================================

Architecture:
- Also known as BN-Inception (Batch Normalization Inception)
- Introduces batch normalization to Inception modules
- Factorizes convolutions: nxn -> 1xn and nx1 (e.g., 3x3 -> 1x3 + 3x1)
- Reduces internal covariate shift
- Allows higher learning rates and faster convergence

Key Features:
- Batch normalization after every convolution
- Factorized convolutions for computational efficiency
- Reduces representational bottleneck
- Better gradient propagation through the network
- More stable training compared to V1

Input:
- Shape: (batch_size, 3, 224, 224)
- Color: RGB images
- Normalization: Typically normalized to [0, 1] or [-1, 1]

Output:
- Shape: (batch_size, num_classes)
- Logits for each class (before softmax)

Computational Complexity:
- Parameters: ~11M (depending on configuration)
- FLOPs: ~2-3 GFLOPs

Example Usage:
    model = Pytorch_InceptionV2()
    model.init_device('cuda').init_model(num_classes=1000)

    # Forward pass
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model.get_model()(input_tensor)
    # output shape: (1, 1000)
"""

import torch
import torch.nn as nn

class InceptionV2Block(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3dbl_red, ch3x3dbl, pool_proj):
        super(InceptionV2Block, self).__init__()

        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv -> 3x3 conv branch (factorized to 1x3 and 3x1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3, ch3x3, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv -> 3x3 conv -> 3x3 conv branch (double 3x3 factorized)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3dbl_red, kernel_size=1),
            nn.BatchNorm2d(ch3x3dbl_red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3dbl_red, ch3x3dbl, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(ch3x3dbl),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3dbl, ch3x3dbl, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(ch3x3dbl),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3dbl, ch3x3dbl, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(ch3x3dbl),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3dbl, ch3x3dbl, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(ch3x3dbl),
            nn.ReLU(inplace=True)
        )

        # 3x3 avg pooling -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV2, self).__init__()

        # Initial convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Inception blocks
        self.inception3a = InceptionV2Block(192, 64, 64, 64, 64, 96, 32)
        self.inception3b = InceptionV2Block(256, 64, 64, 96, 64, 96, 64)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionV2Block(320, 224, 64, 96, 96, 128, 128)
        self.inception4b = InceptionV2Block(576, 192, 96, 128, 96, 128, 128)
        self.inception4c = InceptionV2Block(576, 160, 128, 160, 128, 160, 128)
        self.inception4d = InceptionV2Block(608, 96, 128, 192, 160, 192, 128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionV2Block(608, 352, 192, 320, 160, 224, 128)
        self.inception5b = InceptionV2Block(1024, 352, 192, 320, 192, 224, 128)

        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

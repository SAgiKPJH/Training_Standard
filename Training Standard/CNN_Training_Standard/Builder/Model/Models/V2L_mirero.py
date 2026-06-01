"""
V2L_mirero (PyTorch)
====================
EfficientNetV2L backbone + GlobalAveragePooling + Dense(softmax).
Mirero 사내 명명 규칙의 EfficientNetV2 Large 변종.

Architecture:
- Backbone: torchvision.models.efficientnet_v2_l
- Output: Linear(num_classes) + Softmax(dim=1)

입력:
- Shape: (batch_size, 3, 480, 480)

주의:
- V2L은 V2M보다 더 크므로 OOM 가능성 있음. batch_size 조정 필요.
"""
import torch
import torch.nn as nn
import torchvision


class V2L_mirero(nn.Module):
    is_mirero_softmax = True

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_l(num_classes=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.backbone(x)
        return self.softmax(logits)

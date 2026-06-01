"""
B7_mirero (PyTorch)
===================
EfficientNetB7 backbone + GlobalAveragePooling + Dense(softmax).
Mirero 사내 명명 규칙의 EfficientNet B7 변종.

Architecture:
- Backbone: torchvision.models.efficientnet_b7
- Output: Linear(num_classes) + Softmax(dim=1)

입력:
- Shape: (batch_size, 3, 600, 600)  # B7 권장 입력 크기

주의:
- B7은 파라미터가 매우 많아 (~66M) OOM 가능성 높음. batch_size를 작게 설정 권장.
"""
import torch
import torch.nn as nn
import torchvision


class B7_mirero(nn.Module):
    is_mirero_softmax = True

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b7(num_classes=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.backbone(x)
        return self.softmax(logits)

"""
V2S_mirero (PyTorch)
====================
EfficientNetV2S backbone + GlobalAveragePooling + Dense(softmax).
Mirero 사내 명명 규칙의 EfficientNetV2 Small 변종.

Architecture:
- Backbone: torchvision.models.efficientnet_v2_s (classifier 부분 교체)
- Output: Linear(num_classes) + Softmax(dim=1)

차이점 (vs efficientnet_v2_s):
- 출력에 Softmax가 적용되어 확률 분포 출력
- TF의 Dense(num_classes, activation="softmax") 패턴과 매칭

입력:
- Shape: (batch_size, 3, 384, 384)  # V2S 권장 입력 크기

출력:
- Shape: (batch_size, num_classes)
"""
import torch
import torch.nn as nn
import torchvision


class V2S_mirero(nn.Module):
    is_mirero_softmax = True

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(num_classes=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.backbone(x)
        return self.softmax(logits)

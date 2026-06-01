"""
B3_mirero (PyTorch)
===================
EfficientNetB3 backbone + GlobalAveragePooling + Dense(softmax).
Mirero 사내 명명 규칙의 EfficientNet B3 변종.

Architecture:
- Backbone: torchvision.models.efficientnet_b3 (classifier 부분 교체)
- Pooling: AdaptiveAvgPool2d (이미 efficientnet_b3에 내장)
- Output: Linear(num_classes) + Softmax(dim=1)

차이점 (vs efficientnet_b3):
- 출력에 Softmax가 적용되어 확률 분포 출력
- TF의 Dense(num_classes, activation="softmax") 패턴과 매칭

입력:
- Shape: (batch_size, 3, 300, 300)  # B3 권장 입력 크기

출력:
- Shape: (batch_size, num_classes)
- 각 클래스의 확률 (sum = 1.0)
"""
import torch
import torch.nn as nn
import torchvision


class B3_mirero(nn.Module):
    is_mirero_softmax = True  # TrainingBuilder가 감지하여 _LogNLLLoss로 전환

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b3(num_classes=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.backbone(x)
        return self.softmax(logits)

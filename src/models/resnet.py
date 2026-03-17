import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .base import BaseModel


class ResNet18(BaseModel):
    NAME = "resnet18"

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self._backbone = backbone

    def forward(self, x):
        return self._backbone(x)

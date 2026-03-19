import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from .base import BaseModel


class MobileNetV3Large(BaseModel):
    NAME = "mobilenet_v3_large"

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self._backbone = backbone

    def forward(self, x):
        return self._backbone(x)

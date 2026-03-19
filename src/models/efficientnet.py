import torch.nn as nn
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

from .base import BaseModel


class EfficientNetV2L(BaseModel):
    NAME = "efficientnet_v2_l"

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self._backbone = backbone

    def forward(self, x):
        return self._backbone(x)

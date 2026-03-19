import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights

from .base import BaseModel


class ViTB16(BaseModel):
    NAME = "vit_b_16"

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        backbone.heads.head = nn.Linear(backbone.heads.head.in_features, num_classes)
        self._backbone = backbone

    def forward(self, x):
        return self._backbone(x)


class ViTL16(BaseModel):
    NAME = "vit_l_16"

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        backbone.heads.head = nn.Linear(backbone.heads.head.in_features, num_classes)
        self._backbone = backbone

    def forward(self, x):
        return self._backbone(x)

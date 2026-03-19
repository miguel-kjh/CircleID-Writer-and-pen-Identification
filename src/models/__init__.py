from .base import BaseModel
from .efficientnet import EfficientNetV2L
from .mobilenet import MobileNetV3Large
from .resnet import ResNet18
from .vit import ViTB16, ViTL16

_REGISTRY: dict[str, type[BaseModel]] = {
    ResNet18.NAME: ResNet18,
    EfficientNetV2L.NAME: EfficientNetV2L,
    MobileNetV3Large.NAME: MobileNetV3Large,
    ViTB16.NAME: ViTB16,
    ViTL16.NAME: ViTL16,
}


def build_model(name: str, num_classes: int) -> BaseModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](num_classes)

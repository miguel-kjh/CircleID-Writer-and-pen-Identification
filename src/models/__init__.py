from .base import BaseModel
from .efficientnet import EfficientNetV2L
from .mobilenet import MobileNetV3Large
from .resnet import ResNet18

_REGISTRY: dict[str, type[BaseModel]] = {
    ResNet18.NAME: ResNet18,
    EfficientNetV2L.NAME: EfficientNetV2L,
    MobileNetV3Large.NAME: MobileNetV3Large,
}


def build_model(name: str, num_classes: int) -> BaseModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](num_classes)

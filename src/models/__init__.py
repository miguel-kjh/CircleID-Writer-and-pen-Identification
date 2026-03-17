from .base import BaseModel
from .resnet import ResNet18

_REGISTRY: dict[str, type[BaseModel]] = {
    ResNet18.NAME: ResNet18,
}


def build_model(name: str, num_classes: int) -> BaseModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](num_classes)

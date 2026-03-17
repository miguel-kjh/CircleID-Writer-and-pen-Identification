import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes: int) -> nn.Module:
    """ResNet18 backbone with a linear classifier head."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

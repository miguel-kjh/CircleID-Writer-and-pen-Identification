import abc
import torch.nn as nn


class BaseModel(nn.Module, abc.ABC):
    """Abstract base class for all backbone models."""

    NAME: str = ""  # Subclasses must override

    @classmethod
    def get_name(cls) -> str:
        return cls.NAME

    def __init__(self) -> None:
        super().__init__()

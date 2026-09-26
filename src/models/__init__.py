"""Model adapters and configuration registry."""

from .base_adapter import ModelAdapter
from .fire_adapter import FireAdapter
from .model_registry import ModelDefinition, ModelRegistry
from .yolo_adapter import YoloAdapter

__all__ = [
    "FireAdapter",
    "ModelAdapter",
    "ModelDefinition",
    "ModelRegistry",
    "YoloAdapter",
]

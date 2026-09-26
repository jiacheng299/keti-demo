"""YOLO-backed adapter for the fire scene plugin."""

from .yolo_adapter import YoloAdapter


class FireAdapter(YoloAdapter):
    """Use the shared YOLO lifecycle with fire-model configuration."""

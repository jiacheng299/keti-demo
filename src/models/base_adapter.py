"""Common interface implemented by all local vision model adapters."""

from typing import Any, Protocol, runtime_checkable

from src.schemas.detection import Detection


@runtime_checkable
class ModelAdapter(Protocol):
    """Lifecycle and prediction contract for a model plugin."""

    def load(self) -> None:
        """Load model resources before inference."""

    def predict(self, frame: Any) -> list[Detection]:
        """Return normalized detections for one video frame."""

    def unload(self) -> None:
        """Release model resources after inference."""

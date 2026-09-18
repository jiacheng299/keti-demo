"""Shared validated data contracts for the demo pipeline."""

from .detection import Detection
from .event import Event
from .scene_spec import AlertSpec, RuleSpec, SceneSpec

__all__ = ["AlertSpec", "Detection", "Event", "RuleSpec", "SceneSpec"]

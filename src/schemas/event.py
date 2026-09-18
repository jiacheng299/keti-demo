"""Normalized event record exported by every scene plugin."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .scene_spec import SceneType


class Event(BaseModel):
    """Evidence record shared by UI, JSON, CSV, and snapshots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    scene_type: SceneType
    event_type: str = Field(min_length=1)
    target_class: str = Field(min_length=1)
    track_id: int | None = Field(default=None, ge=0)
    timestamp_seconds: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    trigger_rule: str = Field(min_length=1)
    snapshot_path: str | None = None
    alert_status: Literal["suspected", "confirmed"]
    model_id: str = Field(min_length=1)

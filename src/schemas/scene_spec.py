"""Validated configuration passed from a template or language model to the pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SceneType = Literal["border", "fire"]
RuleType = Literal[
    "object_present",
    "enter_region",
    "leave_region",
    "dwell",
    "count_greater_than",
    "consecutive_frames",
    "confidence_greater_than",
    "and",
    "or",
]


class RuleSpec(BaseModel):
    """One allowlisted rule and its rule-specific parameters."""

    model_config = ConfigDict(extra="forbid")

    type: RuleType
    params: dict[str, Any] = Field(default_factory=dict)


class AlertSpec(BaseModel):
    """Shared alert behavior independent of the selected scene plugin."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    cooldown_seconds: float = Field(default=5.0, ge=0, le=3600)


class SceneSpec(BaseModel):
    """Restricted instruction set consumed by the local video pipeline."""

    model_config = ConfigDict(extra="forbid")

    scene_type: SceneType
    model_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_-]+$")
    targets: list[str] = Field(min_length=1, max_length=20)
    rules: list[RuleSpec] = Field(min_length=1, max_length=20)
    alert: AlertSpec = Field(default_factory=AlertSpec)

    @field_validator("targets")
    @classmethod
    def validate_targets(cls, targets: list[str]) -> list[str]:
        if any(not target.strip() for target in targets):
            raise ValueError("targets must not contain blank values")
        if len(set(targets)) != len(targets):
            raise ValueError("targets must not contain duplicates")
        return targets

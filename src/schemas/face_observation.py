"""Measured facial geometry, not a classification of a person's mental state."""

from pydantic import BaseModel, ConfigDict, Field


class FaceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool = False
    track_id: int | None = Field(default=None, ge=0)
    reason: str = "未检测到人脸"
    bbox_xyxy: tuple[float, float, float, float] | None = None
    left_ear: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    right_ear: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    yaw_degrees: float = Field(default=0, allow_inf_nan=False)
    eye_points: tuple[tuple[float, float], ...] = ()
    continuous: bool = True

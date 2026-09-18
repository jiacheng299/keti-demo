"""Normalized output produced by every local model adapter."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Detection(BaseModel):
    """One detected object in one processed video frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: tuple[float, float, float, float]
    track_id: int | None = Field(default=None, ge=0)

    @field_validator("bbox_xyxy")
    @classmethod
    def validate_bbox(
        cls, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must satisfy x2 > x1 and y2 > y1")
        return bbox

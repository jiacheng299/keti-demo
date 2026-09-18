"""Shared frame and event contracts for stateful rules."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.detection import Detection

#当前视频位置
class FrameState(BaseModel):
    """Minimal video position supplied to every rule evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)

#规则候选事件
class EventCandidate(BaseModel):
    """Rule result before event IDs, screenshots, and persistence are added."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    event_type: str = Field(min_length=1)
    target_class: str = Field(min_length=1)
    track_id: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    trigger_rule: str = Field(min_length=1)
    alert_status: Literal["suspected", "confirmed"] = "confirmed"

#规则接口合同
class StatefulRule(Protocol):
    def evaluate(
        self,
        detections: list[Detection],
        frame_state: FrameState,
    ) -> list[EventCandidate]: ...


def detection_center(detection: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = detection.bbox_xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def event_from_detection(
    detection: Detection,
    frame_state: FrameState,
    event_type: str,
    trigger_rule: str,
) -> EventCandidate:
    if detection.track_id is None:
        raise ValueError("rule events require a track_id")
    return EventCandidate(
        frame_id=frame_state.frame_id,
        timestamp_seconds=frame_state.timestamp_seconds,
        event_type=event_type,
        target_class=detection.class_name,
        track_id=detection.track_id,
        confidence=detection.confidence,
        trigger_rule=trigger_rule,
    )

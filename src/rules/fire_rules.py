"""Stateful confirmation rules for fire-scene detections."""

from typing import Literal

from src.schemas.detection import Detection

from .base_rule import EventCandidate, FrameState


class ConsecutiveFramesRule:
    """Emit one suspected and one confirmed event for each fire episode."""

    def __init__(
        self,
        rule_id: str,
        required_frames: int,
        max_gap_frames: int = 0,
    ) -> None:
        if required_frames <= 0:
            raise ValueError("required_frames must be greater than zero")
        if max_gap_frames < 0:
            raise ValueError("max_gap_frames must not be negative")
        self.rule_id = rule_id
        self.required_frames = required_frames
        self.max_gap_frames = max_gap_frames
        self._matching_frames = 0
        self._gap_frames = 0
        self._suspected_emitted = False
        self._confirmed = False

    def evaluate(
        self,
        detections: list[Detection],
        frame_state: FrameState,
    ) -> list[EventCandidate]:
        if not detections:
            self._gap_frames += 1
            if self._gap_frames > self.max_gap_frames:
                self._reset()
            return []

        self._gap_frames = 0
        self._matching_frames += 1
        representative = max(detections, key=lambda item: item.confidence)
        events: list[EventCandidate] = []

        if not self._suspected_emitted:
            events.append(
                self._event(
                    representative,
                    frame_state,
                    event_type="fire_suspected",
                    alert_status="suspected",
                )
            )
            self._suspected_emitted = True

        if not self._confirmed and self._matching_frames >= self.required_frames:
            events.append(
                self._event(
                    representative,
                    frame_state,
                    event_type="fire_confirmed",
                    alert_status="confirmed",
                )
            )
            self._confirmed = True

        return events

    def _reset(self) -> None:
        self._matching_frames = 0
        self._gap_frames = 0
        self._suspected_emitted = False
        self._confirmed = False

    def _event(
        self,
        detection: Detection,
        frame_state: FrameState,
        event_type: str,
        alert_status: Literal["suspected", "confirmed"],
    ) -> EventCandidate:
        return EventCandidate(
            frame_id=frame_state.frame_id,
            timestamp_seconds=frame_state.timestamp_seconds,
            event_type=event_type,
            target_class=detection.class_name,
            track_id=None,
            confidence=detection.confidence,
            trigger_rule=self.rule_id,
            alert_status=alert_status,
        )

"""Stateful time-based rules for tracked detections."""

from collections.abc import Sequence

from src.schemas.detection import Detection

from .base_rule import (
    EventCandidate,
    FrameState,
    detection_center,
    event_from_detection,
)
from .region_rules import Point, normalize_polygon, point_in_polygon


class DwellRule:
    def __init__(
        self,
        rule_id: str,
        polygon: Sequence[Sequence[float]],
        dwell_seconds: float,
    ) -> None:
        if dwell_seconds <= 0:
            raise ValueError("dwell_seconds must be greater than zero")
        self.rule_id = rule_id
        self.polygon: tuple[Point, ...] = normalize_polygon(polygon)
        self.dwell_seconds = dwell_seconds
        self._entered_at: dict[int, float] = {}
        self._triggered_tracks: set[int] = set()

    def evaluate(
        self,
        detections: list[Detection],
        frame_state: FrameState,
    ) -> list[EventCandidate]:
        events: list[EventCandidate] = []
        for detection in detections:
            if detection.track_id is None:
                continue
            track_id = detection.track_id
            inside = point_in_polygon(detection_center(detection), self.polygon)
            if not inside:
                self._entered_at.pop(track_id, None)
                self._triggered_tracks.discard(track_id)
                continue

            entered_at = self._entered_at.setdefault(
                track_id, frame_state.timestamp_seconds
            )
            if (
                track_id not in self._triggered_tracks
                and frame_state.timestamp_seconds - entered_at >= self.dwell_seconds
            ):
                events.append(
                    event_from_detection(
                        detection,
                        frame_state,
                        "dwell",
                        self.rule_id,
                    )
                )
                self._triggered_tracks.add(track_id)
        return events

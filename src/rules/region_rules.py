"""Enter and leave transitions for polygon regions."""

from collections.abc import Sequence

import cv2
import numpy as np

from src.schemas.detection import Detection

from .base_rule import (
    EventCandidate,
    FrameState,
    detection_center,
    event_from_detection,
)


Point = tuple[float, float]


def normalize_polygon(polygon: Sequence[Sequence[float]]) -> tuple[Point, ...]:
    points = tuple((float(point[0]), float(point[1])) for point in polygon)
    if len(points) < 3:
        raise ValueError("polygon must contain at least three points")
    return points


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    contour = np.asarray(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


class _RegionTransitionRule:
    def __init__(
        self,
        rule_id: str,
        polygon: Sequence[Sequence[float]],
        event_type: str,
    ) -> None:
        self.rule_id = rule_id
        self.polygon = normalize_polygon(polygon)
        self.event_type = event_type
        self._inside_by_track: dict[int, bool] = {}

    def evaluate(
        self,
        detections: list[Detection],
        frame_state: FrameState,
    ) -> list[EventCandidate]:
        events: list[EventCandidate] = []
        for detection in detections:
            if detection.track_id is None:
                continue
            inside = point_in_polygon(detection_center(detection), self.polygon)
            previous = self._inside_by_track.get(detection.track_id)
            should_fire = previous is not None and (
                (self.event_type == "enter_region" and not previous and inside)
                or (self.event_type == "leave_region" and previous and not inside)
            )
            self._inside_by_track[detection.track_id] = inside
            if should_fire:
                events.append(
                    event_from_detection(
                        detection,
                        frame_state,
                        self.event_type,
                        self.rule_id,
                    )
                )
        return events


class EnterRegionRule(_RegionTransitionRule):
    def __init__(self, rule_id: str, polygon: Sequence[Sequence[float]]) -> None:
        super().__init__(rule_id, polygon, "enter_region")


class LeaveRegionRule(_RegionTransitionRule):
    def __init__(self, rule_id: str, polygon: Sequence[Sequence[float]]) -> None:
        super().__init__(rule_id, polygon, "leave_region")

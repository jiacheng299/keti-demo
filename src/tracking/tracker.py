"""Minimal same-class center-distance tracker for short demo clips."""

from dataclasses import dataclass
from math import hypot

from src.schemas.detection import Detection


@dataclass
class _TrackState:
    class_name: str
    center: tuple[float, float]
    last_seen_seconds: float


class Tracker:
    """Assign stable IDs by matching nearby same-class detections."""

    def __init__(self, max_distance: float = 80.0, max_age_seconds: float = 1.0):
        if max_distance <= 0:
            raise ValueError("max_distance must be greater than zero")
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be non-negative")
        self.max_distance = max_distance
        self.max_age_seconds = max_age_seconds
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 0
        self._last_timestamp: float | None = None

    def update(
        self,
        detections: list[Detection],
        timestamp_seconds: float,
    ) -> list[Detection]:
        if timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        if self._last_timestamp is not None and timestamp_seconds < self._last_timestamp:
            raise ValueError("timestamp_seconds must not move backwards")
        self._last_timestamp = timestamp_seconds

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if timestamp_seconds - track.last_seen_seconds <= self.max_age_seconds
        }
        available_track_ids = set(self._tracks)
        tracked_detections: list[Detection] = []

        for detection in detections:
            center = self._center(detection)
            track_id = self._nearest_track_id(
                detection.class_name,
                center,
                available_track_ids,
            )
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
            else:
                available_track_ids.remove(track_id)

            self._tracks[track_id] = _TrackState(
                class_name=detection.class_name,
                center=center,
                last_seen_seconds=timestamp_seconds,
            )
            tracked_detections.append(
                detection.model_copy(update={"track_id": track_id})
            )

        return tracked_detections

    @staticmethod
    def _center(detection: Detection) -> tuple[float, float]:
        x1, y1, x2, y2 = detection.bbox_xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _nearest_track_id(
        self,
        class_name: str,
        center: tuple[float, float],
        available_track_ids: set[int],
    ) -> int | None:
        candidates = []
        for track_id in available_track_ids:
            track = self._tracks[track_id]
            if track.class_name != class_name:
                continue
            distance = hypot(center[0] - track.center[0], center[1] - track.center[1])
            if distance <= self.max_distance:
                candidates.append((distance, track_id))
        return min(candidates)[1] if candidates else None

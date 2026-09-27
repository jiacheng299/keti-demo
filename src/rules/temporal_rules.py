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
from .progress import RuleProgress


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
        self.progress: list[RuleProgress] = []

    def evaluate(
        self,
        detections: list[Detection],
        frame_state: FrameState,
    ) -> list[EventCandidate]:
        events: list[EventCandidate] = []
        self.progress = []
        for detection in detections:
            if detection.track_id is None:
                continue
            track_id = detection.track_id
            inside = point_in_polygon(detection_center(detection), self.polygon)
            if not inside:
                self._entered_at.pop(track_id, None)
                self._triggered_tracks.discard(track_id)
                self.progress.append(RuleProgress(self.rule_id,self.rule_id,"目标在区域外，滞留计时归零",track_id=track_id,required=self.dwell_seconds,unit="秒"))
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
            triggered = track_id in self._triggered_tracks
            self.progress.append(RuleProgress(self.rule_id,self.rule_id,"已达滞留阈值，本次停留不重复报警" if triggered else "区域内滞留，尚未达到时长",track_id=track_id,current=frame_state.timestamp_seconds-entered_at,required=self.dwell_seconds,unit="秒",status="confirmed" if triggered else "tracking"))
        if not self.progress:
            self.progress = [RuleProgress(self.rule_id,self.rule_id,"未观察到所选目标；同 ID 重现时按视频时间判断",required=self.dwell_seconds,unit="秒")]
        return events

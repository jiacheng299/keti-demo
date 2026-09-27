"""Region-level occupancy episodes, including a post that starts empty."""

from .base_rule import EventCandidate, detection_center
from .progress import RuleProgress
from .region_rules import normalize_polygon, point_in_polygon


class UnderstaffedRule:
    def __init__(self, rule_id, polygon, min_count=1, seconds=5,
                 recovery_seconds=1, startup_seconds=2):
        self.rule_id = rule_id
        self.polygon = normalize_polygon(polygon)
        self.min_count = min_count
        self.seconds = seconds
        self.recovery_seconds = recovery_seconds
        self.startup_seconds = startup_seconds
        self.started = self.missing_since = self.recovered_since = None
        self.last_time = None
        self.alarmed = False
        self.progress = []

    def evaluate(self, detections, frame_state):
        now = frame_state.timestamp_seconds
        if self.started is None:
            self.started = now
        # A discontinuity must not be counted as observed empty time.
        max_gap = max(0.5, 2 / (frame_state.effective_fps or 10))
        if self.last_time is not None and now - self.last_time > max_gap:
            self.missing_since = self.recovered_since = None
        self.last_time = now
        count = sum(d.class_name == "person" and
                    point_in_polygon(detection_center(d), self.polygon)
                    for d in detections)
        events = []
        current = 0.0
        required = self.seconds
        if now < self.started + self.startup_seconds:
            status, reason = "unknown", f"初始化：{count}/{self.min_count} 人"
        elif count < self.min_count:
            self.recovered_since = None
            if self.missing_since is None:
                self.missing_since = now
            current = now - self.missing_since
            if current + 1e-9 >= self.seconds and not self.alarmed:
                self.alarmed = True
                events.append(self._event("post_unstaffed", frame_state))
            status = "confirmed" if self.alarmed else "tracking"
            reason = f"{'岗位缺员' if self.alarmed else '缺员计时'}：{count}/{self.min_count} 人"
        else:
            self.missing_since = None
            if self.alarmed:
                if self.recovered_since is None:
                    self.recovered_since = now
                current, required = now - self.recovered_since, self.recovery_seconds
                if current + 1e-9 >= self.recovery_seconds:
                    self.alarmed = False
                    events.append(self._event("post_recovered", frame_state))
                status = "confirmed" if self.alarmed else "normal"
                reason = f"{'回岗确认中' if self.alarmed else '已回岗'}：{count}/{self.min_count} 人"
            else:
                status, reason = "normal", f"在岗：{count}/{self.min_count} 人"
        self.progress = [RuleProgress(self.rule_id, "岗位", reason, current=current,
                                      required=required, unit="秒", status=status)]
        return events

    def _event(self, event_type, state):
        return EventCandidate(frame_id=state.frame_id, timestamp_seconds=state.timestamp_seconds,
                              event_type=event_type, target_class="person", trigger_rule=self.rule_id)

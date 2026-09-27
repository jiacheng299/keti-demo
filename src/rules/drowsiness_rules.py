"""Conservative sustained eye closure with explicit unknown observations."""

from .base_rule import EventCandidate
from .progress import RuleProgress


class _SingleFaceEyesClosedRule:
    def __init__(self, rule_id, seconds=2, recovery_seconds=0.5, closed_ear=0.20,
                 open_ear=0.24, min_face_width=100, max_yaw_degrees=30):
        self.rule_id = rule_id
        self.seconds, self.recovery_seconds = seconds, recovery_seconds
        self.closed_ear, self.open_ear = closed_ear, open_ear
        self.min_face_width, self.max_yaw = min_face_width, max_yaw_degrees
        self.closed_since = self.open_since = self.last_time = None
        self.alarmed = False
        self.progress = []

    def evaluate(self, detections, frame_state):
        face, now = frame_state.face, frame_state.timestamp_seconds
        max_gap = max(0.5, 2 / (frame_state.effective_fps or 10))
        gap = self.last_time is not None and now - self.last_time > max_gap
        self.last_time = now
        reason = None
        if face is None or not face.valid:
            reason = face.reason if face else "未检测到人脸"
        elif face.bbox_xyxy is None or face.left_ear is None or face.right_ear is None:
            reason = "眼部数据不完整"
        elif face.bbox_xyxy[2] - face.bbox_xyxy[0] < self.min_face_width:
            reason = "人脸太小，请使用近景视频"
        elif abs(face.yaw_degrees) > self.max_yaw:
            reason = "侧脸角度过大"
        if reason is not None:
            self.closed_since = self.open_since = None
            self.alarmed = False
            self.progress = [RuleProgress(self.rule_id, "眼部状态", "无法判断：" + reason,
                                          track_id=face.track_id if face else None,
                                          required=self.seconds, unit="秒", status="unknown")]
            return []
        if gap or not face.continuous:
            self.closed_since = self.open_since = None
            self.alarmed = False
        events = []
        current, required = 0.0, self.seconds
        ear_text = f"EAR {face.left_ear:.2f}/{face.right_ear:.2f}"
        if face.left_ear < self.closed_ear and face.right_ear < self.closed_ear:
            self.open_since = None
            if self.closed_since is None:
                self.closed_since = now
            current = now - self.closed_since
            if current + 1e-9 >= self.seconds and not self.alarmed:
                self.alarmed = True
                events.append(self._event("drowsiness_suspected", frame_state))
            status = "confirmed" if self.alarmed else "tracking"
            reason = ("疑似打瞌睡" if self.alarmed else "双眼闭合") + " · " + ear_text
        elif face.left_ear > self.open_ear and face.right_ear > self.open_ear:
            self.closed_since = None
            if self.open_since is None:
                self.open_since = now
            current, required = now - self.open_since, self.recovery_seconds
            if self.alarmed and current + 1e-9 >= self.recovery_seconds:
                self.alarmed = False
                events.append(self._event("eyes_reopened", frame_state))
            status = "confirmed" if self.alarmed else "normal"
            reason = ("睁眼恢复中" if self.alarmed else "睁眼") + " · " + ear_text
            if not self.alarmed:
                current = 0
        else:
            # Ambiguous/single-eye evidence never advances continuous closure.
            self.closed_since = self.open_since = None
            status = "confirmed" if self.alarmed else "waiting"
            reason = ("等待双眼稳定睁开" if self.alarmed else "未满足双眼闭合条件") + " · " + ear_text
        self.progress = [RuleProgress(self.rule_id, "眼部状态", reason, current=current,
                                      track_id=face.track_id,
                                      required=required, unit="秒", status=status)]
        return events

    def _event(self, event_type, state):
        # Confirmed means the temporal rule was met, never a sleep diagnosis.
        return EventCandidate(frame_id=state.frame_id, timestamp_seconds=state.timestamp_seconds,
                              track_id=state.face.track_id,
                              event_type=event_type, target_class="person", trigger_rule=self.rule_id)


class EyesClosedRule:
    """Apply shared thresholds to independent, bounded per-track state machines."""

    def __init__(self, rule_id, seconds=2, recovery_seconds=0.5, closed_ear=0.20,
                 open_ear=0.24, min_face_width=100, max_yaw_degrees=30):
        self.rule_id, self.seconds = rule_id, seconds
        self._params = (rule_id, seconds, recovery_seconds, closed_ear, open_ear,
                        min_face_width, max_yaw_degrees)
        self._tracks = {}
        self.progress = []

    def evaluate(self, detections, frame_state):
        faces = frame_state.faces or ((frame_state.face,) if frame_state.face is not None else ())
        ids = [face.track_id for face in faces]
        if len(set(ids)) != len(ids) or (len(faces) > 1 and None in ids):
            raise ValueError("多人眼部观测必须具有独立且不重复的 track_id")
        # A missing person cannot retain a partially completed timer or alarm.
        self._tracks = {key: rule for key, rule in self._tracks.items() if key in ids}
        self.progress = []
        events = []
        for face in faces:
            rule = self._tracks.setdefault(face.track_id, _SingleFaceEyesClosedRule(*self._params))
            events.extend(rule.evaluate([], frame_state.model_copy(update={"face": face, "faces": ()})))
            self.progress.extend(rule.progress)
        if not faces:
            self.progress = [RuleProgress(self.rule_id, "眼部状态", "无法判断：未检测到人脸",
                                          required=self.seconds, unit="秒", status="unknown")]
        return events

"""Build configured rules, evaluate them, and apply alert cooldown."""

from collections.abc import Sequence
from typing import Any

from src.schemas.detection import Detection
from src.schemas.scene_spec import RuleSpec, SceneSpec

from .base_rule import EventCandidate, FrameState, StatefulRule
from .fire_rules import ConsecutiveFramesRule
from .region_rules import EnterRegionRule, LeaveRegionRule
from .temporal_rules import DwellRule
from .progress import RuleProgress
from .occupancy_rules import UnderstaffedRule
from .drowsiness_rules import EyesClosedRule


class RuleEngine:
    def __init__(self) -> None:
        self._scene_signature: str | None = None
        self._rules: list[StatefulRule] = []
        self._last_emitted_at: dict[tuple[str, str, int | None], float] = {}
        self.progress: list[RuleProgress] = []

    def evaluate(
        self,
        scene_spec: SceneSpec,
        detections: list[Detection],
        frame_state: FrameState,
    ) -> list[EventCandidate]:
        signature = scene_spec.model_dump_json()
        if signature != self._scene_signature:
            self._configure(scene_spec, signature)

        target_detections = [
            detection
            for detection in detections
            if detection.class_name in scene_spec.targets
        ]
        candidates = [
            candidate
            for rule in self._rules
            for candidate in rule.evaluate(target_detections, frame_state)
        ]
        self.progress = []
        for spec, rule in zip(scene_spec.rules, self._rules):
            for item in rule.progress:
                if "polygon" in spec.params:
                    suffix = {"dwell": " 滞留", "enter_region": " 进入", "leave_region": " 离开", "region_understaffed": " 值守"}.get(spec.type, "")
                    item.label = spec.params.get("name",item.rule_id) + suffix
                self.progress.append(item)
        if not scene_spec.alert.enabled:
            for item in self.progress:
                item.reason = "报警已关闭；" + item.reason
            return []

        emitted: list[EventCandidate] = []
        cooldown = scene_spec.alert.cooldown_seconds
        for candidate in candidates:
            key = (
                candidate.trigger_rule,
                candidate.event_type,
                candidate.track_id,
            )
            last_emitted = self._last_emitted_at.get(key)
            episode_event = candidate.event_type in {"post_unstaffed", "post_recovered", "drowsiness_suspected", "eyes_reopened"}
            if not episode_event and last_emitted is not None and (
                candidate.timestamp_seconds - last_emitted < cooldown
            ):
                for item in self.progress:
                    if item.rule_id == candidate.trigger_rule and item.track_id == candidate.track_id:
                        item.reason = f"满足条件，但冷却中，剩余 {cooldown-(candidate.timestamp_seconds-last_emitted):.1f} 秒"
                        item.status = "cooldown"
                continue
            self._last_emitted_at[key] = candidate.timestamp_seconds
            emitted.append(candidate)
        return emitted

    def _configure(self, scene_spec: SceneSpec, signature: str) -> None:
        self._rules = [
            self._build_rule(rule_spec, index)
            for index, rule_spec in enumerate(scene_spec.rules)
        ]
        self._last_emitted_at.clear()
        self._scene_signature = signature

    def _build_rule(self, spec: RuleSpec, index: int) -> StatefulRule:
        params = spec.params
        if spec.type == "region_understaffed":
            return UnderstaffedRule(str(params.get("region_id", f"post:{index}")),
                                    self._require_points(params, "polygon", minimum=3),
                                    **{k: params[k] for k in ("min_count", "seconds", "recovery_seconds", "startup_seconds") if k in params})
        if spec.type == "eyes_closed_duration":
            return EyesClosedRule(str(params.get("rule_id", f"eyes:{index}")),
                                  **{k: params[k] for k in ("seconds", "recovery_seconds", "closed_ear", "open_ear", "min_face_width", "max_yaw_degrees") if k in params})
        if spec.type in {"enter_region", "leave_region", "dwell"}:
            polygon = self._require_points(params, "polygon", minimum=3)
            rule_id = str(params.get("region_id", f"{spec.type}:{index}"))
            if spec.type == "enter_region":
                return EnterRegionRule(rule_id, polygon)
            if spec.type == "leave_region":
                return LeaveRegionRule(rule_id, polygon)
            dwell_seconds = float(params.get("seconds", 5.0))
            return DwellRule(rule_id, polygon, dwell_seconds)

        if spec.type == "consecutive_frames":
            rule_id = str(params.get("rule_id", f"consecutive_frames:{index}"))
            required_frames = int(params.get("frames", 3))
            max_gap_frames = int(params.get("max_gap_frames", 0))
            rule = ConsecutiveFramesRule(
                rule_id,
                required_frames=required_frames,
                max_gap_frames=max_gap_frames,
            )
            rule.duration_seconds = params.get("seconds")
            return rule

        raise ValueError(f"rule type is not implemented: {spec.type}")

    @staticmethod
    def _require_points(
        params: dict[str, Any],
        key: str,
        minimum: int,
        maximum: int | None = None,
    ) -> tuple[tuple[float, float], ...]:
        raw_points = params.get(key)
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, str):
            raise ValueError(f"{key} must be a sequence of points")
        if len(raw_points) < minimum or (
            maximum is not None and len(raw_points) > maximum
        ):
            raise ValueError(f"{key} has an invalid number of points")
        try:
            return tuple((float(point[0]), float(point[1])) for point in raw_points)
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(f"{key} must contain numeric x/y pairs") from error

"""Build configured rules, evaluate them, and apply alert cooldown."""

from collections.abc import Sequence
from typing import Any

from src.schemas.detection import Detection
from src.schemas.scene_spec import RuleSpec, SceneSpec

from .base_rule import EventCandidate, FrameState, StatefulRule
from .region_rules import EnterRegionRule, LeaveRegionRule
from .temporal_rules import DwellRule


class RuleEngine:
    def __init__(self) -> None:
        self._scene_signature: str | None = None
        self._rules: list[StatefulRule] = []
        self._last_emitted_at: dict[tuple[str, str, int], float] = {}

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
        if not scene_spec.alert.enabled:
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
            if last_emitted is not None and (
                candidate.timestamp_seconds - last_emitted < cooldown
            ):
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
        if spec.type in {"enter_region", "leave_region", "dwell"}:
            polygon = self._require_points(params, "polygon", minimum=3)
            rule_id = str(params.get("region_id", f"{spec.type}:{index}"))
            if spec.type == "enter_region":
                return EnterRegionRule(rule_id, polygon)
            if spec.type == "leave_region":
                return LeaveRegionRule(rule_id, polygon)
            dwell_seconds = float(params.get("seconds", 5.0))
            return DwellRule(rule_id, polygon, dwell_seconds)

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

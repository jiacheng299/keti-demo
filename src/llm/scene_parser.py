"""Validate DeepSeek scene JSON and load deterministic offline templates."""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Protocol

from pydantic import ValidationError
import yaml

from src.models.model_registry import ModelRegistry, UnknownModelError
from src.schemas.scene_spec import RuleSpec, SceneSpec
from src.schemas.scene_time import resolve_time_rules

from .deepseek_client import DeepSeekAPIError, MissingAPIKeyError
from .prompts import build_scene_messages
from .requirement_guard import (
    ASSESSMENT_PROMPT, RequirementDecision, RequirementRejected,
    validate_decision_mapping, validate_requirement_text,
)


class SceneParseError(ValueError):
    """Raised when two model responses fail local scene validation."""


class UnknownTemplateError(KeyError):
    """Raised when a template ID is not one of the fixed local choices."""


class CompletionClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]]) -> str: ...


TEMPLATE_FILES = {
    "on_duty": "on_duty.yaml",
    "drowsiness": "drowsiness.yaml",
    "border_intrusion": "border_intrusion.yaml",
    "border_person_intrusion": "border_person_intrusion.yaml",
    "border_vehicle_intrusion": "border_vehicle_intrusion.yaml",
    "border_dwell": "border_dwell.yaml",
    "fire_detection": "fire_detection.yaml",
}

MODEL_SCENES = {"yolo_general": {"border", "on_duty"}, "fire_smoke": {"fire"},
                "face_landmarker": {"drowsiness"}}
RULES_BY_SCENE = {
    "on_duty": {"region_understaffed"},
    "drowsiness": {"eyes_closed_duration"},
    "border": {"enter_region", "leave_region", "dwell"},
    "fire": {"consecutive_frames"},
}
PARAM_KEYS = {
    "region_understaffed": {"region_id", "name", "polygon", "min_count", "seconds", "recovery_seconds", "startup_seconds"},
    "eyes_closed_duration": {"rule_id", "seconds", "recovery_seconds", "closed_ear", "open_ear", "min_face_width", "max_yaw_degrees"},
    "enter_region": {"region_id", "name", "polygon"},
    "leave_region": {"region_id", "name", "polygon"},
    "dwell": {"region_id", "name", "polygon", "seconds"},
    "consecutive_frames": {"rule_id", "frames", "seconds", "max_gap_frames"},
}
ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")


class SceneParser:
    """Turn LLM JSON or fixed YAML into an executable, allowlisted SceneSpec."""

    def __init__(
        self,
        client: CompletionClient,
        model_registry: ModelRegistry,
        *,
        template_dir: str | Path = "config/scenes",
    ) -> None:
        self.client = client
        self.model_registry = model_registry
        self.template_dir = Path(template_dir)

    def parse(self, text: str, *, video_fps: float | None = None) -> SceneSpec:
        """Assess intent and capabilities before returning an executable scene."""
        return self.parse_requirement(text, video_fps=video_fps)[0]

    def parse_requirement(self, text: str, *, video_fps: float | None = None,
                          video_size: tuple[int, int] | None = None) -> tuple[SceneSpec, str]:
        text = validate_requirement_text(text)
        messages = build_scene_messages(text, video_fps=video_fps)
        messages[0]["content"] += ASSESSMENT_PROMPT
        if video_size is not None:
            width, height = video_size
            if width <= 1 or height <= 1:
                raise ValueError("视频尺寸无效")
            messages[0]["content"] += (
                f"\nActual video dimensions are {width}x{height}. Override the default 640x480: "
                f"all polygon coordinates must satisfy 0<=x<{width}, 0<=y<{height}. "
                f"Default editable rectangle: [[{width*.25},{height*.25}],"
                f"[{width*.75},{height*.25}],[{width*.75},{height*.75}],[{width*.25},{height*.75}]]."
            )
        last_error = None
        for attempt in range(2):
            content = self.client.complete(messages)
            try:
                decision = RequirementDecision.model_validate_json(content)
                if decision.status != "supported":
                    # Never salvage a configuration from a rejected assessment.
                    raise RequirementRejected(decision.status, decision.reason,
                                              unmet=decision.unmet_requirements,
                                              vlm_eligible=decision.vlm_eligible, vlm_reason=decision.vlm_reason)
                scene = self._validate_payload(decision.scene)
                validate_decision_mapping(decision, text, scene)
                if video_size is not None:
                    for rule in scene.rules:
                        if "polygon" in rule.params and any(x >= width or y >= height for x,y in rule.params["polygon"]):
                            raise ValueError("区域超出上传视频尺寸")
                if any(r.type == "consecutive_frames" and "seconds" in r.params for r in scene.rules):
                    if video_fps is None:
                        raise RequirementRejected("needs_clarification", "请先上传视频，再按视频帧率确定持续检测时间。")
                    scene = resolve_time_rules(scene, video_fps)
                return scene, decision.summary
            except RequirementRejected:
                raise
            except (ValidationError, ValueError, TypeError, UnknownModelError) as error:
                last_error = error
                if attempt == 0:
                    messages[0]["content"] += "\nThe previous response failed local validation. Return the complete assessment envelope with internally consistent requirements and scene; do not skip assessment."
        raise SceneParseError(f"DeepSeek returned two invalid requirement assessments: {last_error}") from last_error

    def parse_or_template(self, text: str, template_id: str) -> SceneSpec:
        """Use one fixed local template when API parsing cannot produce a scene."""
        try:
            return self.parse(text)
        except (MissingAPIKeyError, DeepSeekAPIError, SceneParseError):
            return self.load_template(template_id)

    def load_template(self, template_id: str) -> SceneSpec:
        """Load a template by fixed ID; never derive a path from user text."""
        try:
            filename = TEMPLATE_FILES[template_id]
        except KeyError as error:
            raise UnknownTemplateError(f"unknown template: {template_id}") from error

        path = self.template_dir / filename
        with path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file)
        try:
            return self._validate_payload(payload)
        except (ValidationError, UnknownModelError, TypeError, ValueError) as error:
            raise SceneParseError(f"invalid offline template '{template_id}': {error}") from error

    def _validate_payload(self, payload: object) -> SceneSpec:
        scene = SceneSpec.model_validate(payload)
        definition = self.model_registry.get(scene.model_id)

        expected_scene = MODEL_SCENES.get(scene.model_id)
        if scene.scene_type not in (expected_scene or set()):
            raise ValueError(
                f"model '{scene.model_id}' does not belong to scene_type '{scene.scene_type}'"
            )

        unsupported_targets = sorted(set(scene.targets) - set(definition.classes))
        if unsupported_targets:
            raise ValueError(f"unsupported targets for '{scene.model_id}': {unsupported_targets}")

        allowed_rules = RULES_BY_SCENE[scene.scene_type]
        if scene.scene_type in {"on_duty", "drowsiness"} and scene.targets != ["person"]:
            raise ValueError("在岗和打瞌睡监测的目标必须为人员")
        if scene.scene_type == "drowsiness" and len(scene.rules) != 1:
            raise ValueError("打瞌睡监测使用一组共享阈值规则，对每个人分别计时")
        identifiers = set()
        for rule in scene.rules:
            if rule.type not in allowed_rules:
                raise ValueError(
                    f"rule '{rule.type}' is not supported for {scene.scene_type} scenes"
                )
            self._validate_rule(rule)
            identifier = rule.params.get("region_id", rule.params.get("rule_id"))
            if scene.scene_type == "on_duty" and identifier in identifiers:
                raise ValueError("每个岗位必须使用不同的 region_id")
            identifiers.add(identifier)
        return scene

    def _validate_rule(self, rule: RuleSpec) -> None:
        if "name" in rule.params and (not isinstance(rule.params["name"], str) or not 1 <= len(rule.params["name"].strip()) <= 40):
            raise ValueError("区域名称须为 1–40 个字符")
        extra_keys = set(rule.params) - PARAM_KEYS[rule.type]
        if extra_keys:
            raise ValueError(
                f"unsupported parameters for '{rule.type}': {sorted(extra_keys)}"
            )

        identifier_key = "rule_id" if rule.type in {"consecutive_frames", "eyes_closed_duration"} else "region_id"
        identifier = rule.params.get(identifier_key)
        if identifier is not None and (
            not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier)
        ):
            raise ValueError(f"{identifier_key} must use lowercase letters, digits, _ or -")

        if rule.type in {"enter_region", "leave_region", "dwell", "region_understaffed"}:
            self._validate_polygon(rule.params.get("polygon"))

        if rule.type == "dwell" and "seconds" in rule.params:
            self._require_number(rule.params["seconds"], "seconds", minimum=0.1, maximum=3600)

        if rule.type in {"region_understaffed", "eyes_closed_duration"}:
            for key in ("seconds", "recovery_seconds"):
                if key in rule.params:
                    self._require_number(rule.params[key], key, minimum=0.1, maximum=3600)
        if rule.type == "region_understaffed":
            self._require_integer(rule.params.get("min_count", 1), "min_count", minimum=1, maximum=100)
            self._require_number(rule.params.get("startup_seconds", 2), "startup_seconds", minimum=0, maximum=3600)
        if rule.type == "eyes_closed_duration":
            for key, default in (("closed_ear", 0.20), ("open_ear", 0.24)):
                self._require_number(rule.params.get(key, default), key, minimum=0.01, maximum=0.8)
            if rule.params.get("closed_ear", 0.20) >= rule.params.get("open_ear", 0.24):
                raise ValueError("睁眼阈值必须高于闭眼阈值")
            self._require_integer(rule.params.get("min_face_width", 100), "min_face_width", minimum=40, maximum=2000)
            self._require_number(rule.params.get("max_yaw_degrees", 30), "max_yaw_degrees", minimum=5, maximum=45)

        if rule.type == "consecutive_frames":
            if "seconds" in rule.params:
                self._require_number(rule.params["seconds"], "seconds", minimum=0.1, maximum=3600)
            if "frames" in rule.params:
                self._require_integer(rule.params["frames"], "frames", minimum=1, maximum=1000000)
            if "max_gap_frames" in rule.params:
                self._require_integer(
                    rule.params["max_gap_frames"],
                    "max_gap_frames",
                    minimum=0,
                    maximum=100,
                )

    @staticmethod
    def _validate_polygon(value: object) -> None:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or not 3 <= len(value) <= 20
        ):
            raise ValueError("polygon must contain between 3 and 20 points")

        for point in value:
            if (
                not isinstance(point, Sequence)
                or isinstance(point, (str, bytes))
                or len(point) != 2
            ):
                raise ValueError("polygon points must contain exactly two coordinates")
            for coordinate in point:
                if (
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, (int, float))
                    or not 0 <= coordinate <= 10000
                ):
                    raise ValueError("polygon coordinates must be between 0 and 10000")

    @staticmethod
    def _require_number(value: object, name: str, *, minimum: float, maximum: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name} must be between {minimum} and {maximum}")

    @staticmethod
    def _require_integer(value: object, name: str, *, minimum: int, maximum: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")

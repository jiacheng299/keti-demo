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

from .deepseek_client import DeepSeekAPIError, MissingAPIKeyError
from .prompts import build_scene_messages


class SceneParseError(ValueError):
    """Raised when two model responses fail local scene validation."""


class UnknownTemplateError(KeyError):
    """Raised when a template ID is not one of the fixed local choices."""


class CompletionClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]]) -> str: ...


TEMPLATE_FILES = {
    "border_person_intrusion": "border_person_intrusion.yaml",
    "border_vehicle_intrusion": "border_vehicle_intrusion.yaml",
    "border_dwell": "border_dwell.yaml",
    "fire_detection": "fire_detection.yaml",
}

MODEL_SCENES = {"yolo_general": "border", "fire_smoke": "fire"}
RULES_BY_SCENE = {
    "border": {"enter_region", "leave_region", "dwell"},
    "fire": {"consecutive_frames"},
}
PARAM_KEYS = {
    "enter_region": {"region_id", "polygon"},
    "leave_region": {"region_id", "polygon"},
    "dwell": {"region_id", "polygon", "seconds"},
    "consecutive_frames": {"rule_id", "frames", "max_gap_frames"},
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

    def parse(self, text: str) -> SceneSpec:
        """Request and validate a scene, retrying one invalid model output."""
        messages = build_scene_messages(text)
        last_error: Exception | None = None
        for _attempt in range(2):
            content = self.client.complete(messages)
            try:
                payload = json.loads(content)
                return self._validate_payload(payload)
            except (
                json.JSONDecodeError,
                ValidationError,
                UnknownModelError,
                TypeError,
                ValueError,
            ) as error:
                last_error = error

        raise SceneParseError(
            f"DeepSeek returned two invalid scene configurations: {last_error}"
        ) from last_error

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
        if expected_scene != scene.scene_type:
            raise ValueError(
                f"model '{scene.model_id}' does not belong to scene_type '{scene.scene_type}'"
            )

        unsupported_targets = sorted(set(scene.targets) - set(definition.classes))
        if unsupported_targets:
            raise ValueError(f"unsupported targets for '{scene.model_id}': {unsupported_targets}")

        allowed_rules = RULES_BY_SCENE[scene.scene_type]
        for rule in scene.rules:
            if rule.type not in allowed_rules:
                raise ValueError(
                    f"rule '{rule.type}' is not supported for {scene.scene_type} scenes"
                )
            self._validate_rule(rule)
        return scene

    def _validate_rule(self, rule: RuleSpec) -> None:
        extra_keys = set(rule.params) - PARAM_KEYS[rule.type]
        if extra_keys:
            raise ValueError(
                f"unsupported parameters for '{rule.type}': {sorted(extra_keys)}"
            )

        identifier_key = "rule_id" if rule.type == "consecutive_frames" else "region_id"
        identifier = rule.params.get(identifier_key)
        if identifier is not None and (
            not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier)
        ):
            raise ValueError(f"{identifier_key} must use lowercase letters, digits, _ or -")

        if rule.type in {"enter_region", "leave_region", "dwell"}:
            self._validate_polygon(rule.params.get("polygon"))

        if rule.type == "dwell" and "seconds" in rule.params:
            self._require_number(rule.params["seconds"], "seconds", minimum=0.1, maximum=3600)

        if rule.type == "consecutive_frames":
            if "frames" in rule.params:
                self._require_integer(rule.params["frames"], "frames", minimum=1, maximum=1000)
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

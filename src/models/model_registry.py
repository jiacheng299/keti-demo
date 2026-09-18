"""Safe configuration lookup and allowlisted construction of model adapters."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .base_adapter import ModelAdapter


class UnknownModelError(KeyError):
    """Raised when a SceneSpec requests a model ID absent from the registry."""


class AdapterNotRegisteredError(LookupError):
    """Raised when a model is known but its adapter factory is unavailable."""


class ModelDefinition(BaseModel):
    """Validated, inert model metadata loaded from YAML."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    adapter: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    weights: Path
    classes: tuple[str, ...] = Field(min_length=1)
    device: str = Field(default="cpu", pattern=r"^cpu$")
    imgsz: int = Field(default=416, ge=160, le=1280)
    confidence: float = Field(default=0.25, ge=0, le=1)


AdapterFactory = Callable[[ModelDefinition], ModelAdapter]


class ModelRegistry:
    """Read model metadata and construct adapters through explicit factories."""

    def __init__(self, definitions: Mapping[str, ModelDefinition]) -> None:
        self._definitions = dict(definitions)
        self._factories: dict[str, AdapterFactory] = {}

    @classmethod
    def from_config(cls, path: str | Path) -> "ModelRegistry":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file)

        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("model registry config must declare version: 1")
        raw_models = payload.get("models")
        if not isinstance(raw_models, dict) or not raw_models:
            raise ValueError("model registry config must contain at least one model")

        definitions = {
            model_id: ModelDefinition.model_validate(
                {"model_id": model_id, **cls._require_mapping(model_id, raw_model)}
            )
            for model_id, raw_model in raw_models.items()
        }
        return cls(definitions)

    @staticmethod
    def _require_mapping(model_id: str, raw_model: Any) -> dict[str, Any]:
        if not isinstance(raw_model, dict):
            raise ValueError(f"model '{model_id}' definition must be a mapping")
        return raw_model

    def register_factory(self, adapter_name: str, factory: AdapterFactory) -> None:
        """Allow one trusted adapter implementation to be constructed by name."""

        if not adapter_name or adapter_name not in {
            definition.adapter for definition in self._definitions.values()
        }:
            raise ValueError(f"adapter '{adapter_name}' is not used by this registry")
        self._factories[adapter_name] = factory

    def get(self, model_id: str) -> ModelDefinition:
        try:
            return self._definitions[model_id]
        except KeyError as error:
            raise UnknownModelError(f"unknown model_id: {model_id}") from error

    def create(self, model_id: str) -> ModelAdapter:
        definition = self.get(model_id)
        try:
            factory = self._factories[definition.adapter]
        except KeyError as error:
            raise AdapterNotRegisteredError(
                f"adapter factory is not registered: {definition.adapter}"
            ) from error
        return factory(definition)

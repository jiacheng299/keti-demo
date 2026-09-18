from pathlib import Path

import pytest

from src.models.model_registry import ModelRegistry, UnknownModelError


class StubAdapter:
    def __init__(self, definition):
        self.definition = definition

    def load(self) -> None:
        return None

    def predict(self, frame) -> list:
        return []

    def unload(self) -> None:
        return None


def write_model_config(path: Path) -> None:
    path.write_text(
        """
version: 1
models:
  test_detector:
    adapter: stub
    weights: models/test/model.pt
    classes: [person, car]
    device: cpu
    imgsz: 416
""".strip(),
        encoding="utf-8",
    )


def test_registry_rejects_an_unknown_model_id(tmp_path):
    config_path = tmp_path / "models.yaml"
    write_model_config(config_path)
    registry = ModelRegistry.from_config(config_path)

    with pytest.raises(UnknownModelError, match="missing_detector"):
        registry.create("missing_detector")


def test_registry_creates_a_known_model_with_an_allowlisted_factory(tmp_path):
    config_path = tmp_path / "models.yaml"
    write_model_config(config_path)
    registry = ModelRegistry.from_config(config_path)
    registry.register_factory("stub", StubAdapter)

    adapter = registry.create("test_detector")

    assert isinstance(adapter, StubAdapter)
    assert adapter.definition.model_id == "test_detector"
    assert adapter.definition.classes == ("person", "car")
    assert adapter.definition.device == "cpu"

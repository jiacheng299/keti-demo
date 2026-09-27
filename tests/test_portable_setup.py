import hashlib
import io
from pathlib import Path

import pytest

from scripts.download_models import ensure_model, model_path
from src.models.model_registry import ModelRegistry


def asset(data=b"verified model"):
    return {"id": "test", "path": "models/test/model.pt", "url": "https://example.org/model.pt",
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def test_download_is_verified_and_second_run_does_not_use_network(tmp_path):
    calls = []
    def opener(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(b"verified model")
    target = ensure_model(asset(), tmp_path, opener)
    assert target.read_bytes() == b"verified model"
    assert ensure_model(asset(), tmp_path, opener) == target
    assert len(calls) == 1
    assert not list(tmp_path.rglob("*.download"))


def test_bad_download_does_not_destroy_existing_model_or_leave_partial_file(tmp_path):
    target = model_path(asset(), tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing model")
    with pytest.raises(ValueError, match="checksum"):
        ensure_model(asset(), tmp_path, lambda *args, **kwargs: io.BytesIO(b"corrupt model!"))
    assert target.read_bytes() == b"existing model"
    assert not list(tmp_path.rglob("*.download"))


def test_failed_download_can_be_retried_without_a_partial_model(tmp_path):
    attempts = []
    def opener(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("network disconnected")
        return io.BytesIO(b"verified model")
    assert ensure_model(asset(), tmp_path, opener).read_bytes() == b"verified model"
    assert len(attempts) == 2


def test_download_manifest_cannot_escape_models_directory(tmp_path):
    item = asset()
    item["path"] = "../outside.pt"
    with pytest.raises(ValueError, match="models directory"):
        model_path(item, tmp_path)


def test_project_weights_are_independent_of_shell_working_directory(tmp_path, monkeypatch):
    root = tmp_path / "project with spaces"
    (root / "config").mkdir(parents=True)
    config = root / "config/models.yaml"
    config.write_text("version: 1\nmodels:\n  detector:\n    adapter: yolo\n    weights: models/border/model.pt\n    classes: [person]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    model = ModelRegistry.from_config(config).get("detector")
    assert model.weights == root / "models/border/model.pt"


def test_absolute_custom_weights_remain_unchanged(tmp_path):
    weight = (tmp_path / "custom.pt").as_posix()
    config = tmp_path / "models.yaml"
    config.write_text(f"version: 1\nmodels:\n  detector:\n    adapter: yolo\n    weights: '{weight}'\n    classes: [person]\n", encoding="utf-8")
    assert ModelRegistry.from_config(config).get("detector").weights == Path(weight)

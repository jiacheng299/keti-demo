import numpy as np

from src.models.fire_adapter import FireAdapter
from src.models.model_registry import ModelDefinition


class FakeBoxes:
    xyxy = np.array(
        [
            [10.0, 20.0, 30.0, 50.0],
            [40.0, 10.0, 60.0, 35.0],
            [5.0, 5.0, 15.0, 15.0],
        ]
    )
    conf = np.array([0.91, 0.70, 0.99])
    cls = np.array([1.0, 0.0, 2.0])


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "smoke", 1: "fire", 2: "person"}


class FakeFireModel:
    names = FakeResult.names

    def predict(self, **options):
        assert options["device"] == "cpu"
        assert options["classes"] == [0, 1]
        return [FakeResult()]


def test_fire_adapter_returns_only_configured_fire_classes():
    definition = ModelDefinition(
        model_id="fire_smoke",
        adapter="fire",
        weights="models/fire/fire_smoke_yolov8n.pt",
        classes=("smoke", "fire"),
        device="cpu",
        imgsz=416,
        confidence=0.25,
    )
    adapter = FireAdapter(definition, model=FakeFireModel())
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    detections = adapter.predict(frame, frame_id=9)

    assert [(item.class_name, item.frame_id) for item in detections] == [
        ("fire", 9),
        ("smoke", 9),
    ]
    assert all(item.track_id is None for item in detections)

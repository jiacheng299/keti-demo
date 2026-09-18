import numpy as np

from src.models.model_registry import ModelDefinition
from src.models.yolo_adapter import YoloAdapter


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.array(
            [
                [10.0, 20.0, 30.0, 50.0],
                [40.0, 10.0, 60.0, 35.0],
                [5.0, 5.0, 15.0, 15.0],
            ]
        )
        self.conf = np.array([0.91, 0.20, 0.99])
        self.cls = np.array([0.0, 2.0, 16.0])


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "person", 2: "car", 16: "dog"}


class FakeYoloModel:
    names = FakeResult.names

    def predict(self, **options):
        assert options["device"] == "cpu"
        assert options["imgsz"] == 416
        assert options["conf"] == 0.25
        assert options["classes"] == [0, 2]
        return [FakeResult()]


def test_yolo_adapter_normalizes_and_filters_detector_output():
    definition = ModelDefinition(
        model_id="test_general",
        adapter="yolo",
        weights="models/test.pt",
        classes=("person", "car"),
        device="cpu",
        imgsz=416,
        confidence=0.25,
    )
    adapter = YoloAdapter(definition, model=FakeYoloModel())
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    detections = adapter.predict(frame, frame_id=7)

    assert [detection.model_dump() for detection in detections] == [
        {
            "frame_id": 7,
            "class_name": "person",
            "confidence": 0.91,
            "bbox_xyxy": (10.0, 20.0, 30.0, 50.0),
            "track_id": None,
        }
    ]

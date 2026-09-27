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


def test_all_vehicle_types_are_inferred_and_normalized_before_rules():
    class Boxes:
        xyxy = np.tile([1., 2., 10., 20.], (7, 1))
        conf = np.array([.9, .9, .9, .9, .9, .99, .1])
        cls = np.array([0, 2, 3, 5, 7, 16, 7])

    class Result:
        names = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 16: "dog"}
        boxes = Boxes()

    class Model:
        names = Result.names

        def predict(self, **options):
            assert options["classes"] == [0, 2, 3, 5, 7]
            return [Result()]

    definition = ModelDefinition(model_id="test", adapter="yolo", weights="unused.pt", classes=("person", "car"))
    detections = YoloAdapter(definition, model=Model()).predict(np.zeros((32,32,3), dtype=np.uint8))
    assert [d.class_name for d in detections] == ["person", "car", "car", "car", "car"]

    # A scene selecting car must also trigger on the model's bus/truck outputs.
    from src.rules.base_rule import FrameState
    from src.rules.rule_engine import RuleEngine
    from src.schemas.scene_spec import SceneSpec
    scene = SceneSpec(scene_type="border", model_id="yolo_general", targets=["car"], rules=[{"type":"dwell", "params":{"polygon":[[0,0],[30,0],[30,30],[0,30]], "seconds":1}}])
    tracked = [d.model_copy(update={"track_id":i}) for i,d in enumerate(detections)]
    engine = RuleEngine()
    engine.evaluate(scene,tracked,FrameState(frame_id=0,timestamp_seconds=0))
    events = engine.evaluate(scene,tracked,FrameState(frame_id=1,timestamp_seconds=1))
    assert len(events) == 4
    assert {e.target_class for e in events} == {"car"}

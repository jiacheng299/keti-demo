from src.rules.base_rule import FrameState
from src.rules.rule_engine import RuleEngine
from src.rules.temporal_rules import DwellRule
from src.schemas.detection import Detection
from src.schemas.scene_spec import SceneSpec


SQUARE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def detection(frame_id: int, center: tuple[float, float]) -> Detection:
    x, y = center
    return Detection(
        frame_id=frame_id,
        class_name="person",
        confidence=0.9,
        bbox_xyxy=(x - 5, y - 5, x + 5, y + 5),
        track_id=2,
    )


def test_dwell_fires_once_after_the_duration_threshold():
    rule = DwellRule("dwell-zone", SQUARE, dwell_seconds=2.0)

    assert rule.evaluate(
        [detection(0, (50, 50))],
        FrameState(frame_id=0, timestamp_seconds=0.0),
    ) == []
    assert rule.evaluate(
        [detection(1, (55, 50))],
        FrameState(frame_id=1, timestamp_seconds=1.9),
    ) == []
    events = rule.evaluate(
        [detection(2, (60, 50))],
        FrameState(frame_id=2, timestamp_seconds=2.0),
    )
    repeated = rule.evaluate(
        [detection(3, (65, 50))],
        FrameState(frame_id=3, timestamp_seconds=3.0),
    )

    assert len(events) == 1
    assert events[0].event_type == "dwell"
    assert repeated == []


def test_rule_engine_suppresses_repeated_events_during_cooldown():
    scene = SceneSpec.model_validate(
        {
            "scene_type": "border",
            "model_id": "yolo_general",
            "targets": ["person"],
            "rules": [
                {
                    "type": "enter_region",
                    "params": {"region_id": "restricted", "polygon": SQUARE},
                }
            ],
            "alert": {"enabled": True, "cooldown_seconds": 5},
        }
    )
    engine = RuleEngine()

    engine.evaluate(
        scene,
        [detection(0, (-10, 50))],
        FrameState(frame_id=0, timestamp_seconds=0.0),
    )
    first = engine.evaluate(
        scene,
        [detection(1, (20, 50))],
        FrameState(frame_id=1, timestamp_seconds=1.0),
    )
    engine.evaluate(
        scene,
        [detection(2, (-10, 50))],
        FrameState(frame_id=2, timestamp_seconds=2.0),
    )
    suppressed = engine.evaluate(
        scene,
        [detection(3, (20, 50))],
        FrameState(frame_id=3, timestamp_seconds=3.0),
    )
    engine.evaluate(
        scene,
        [detection(7, (-10, 50))],
        FrameState(frame_id=7, timestamp_seconds=7.0),
    )
    after_cooldown = engine.evaluate(
        scene,
        [detection(8, (20, 50))],
        FrameState(frame_id=8, timestamp_seconds=8.0),
    )

    assert len(first) == 1
    assert suppressed == []
    assert len(after_cooldown) == 1

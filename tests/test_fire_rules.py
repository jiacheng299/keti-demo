import pytest

from src.rules.base_rule import FrameState
from src.rules.fire_rules import ConsecutiveFramesRule
from src.rules.rule_engine import RuleEngine
from src.schemas.detection import Detection
from src.schemas.scene_spec import SceneSpec


def fire_detection(frame_id: int, confidence: float = 0.9) -> Detection:
    return Detection(
        frame_id=frame_id,
        class_name="fire",
        confidence=confidence,
        bbox_xyxy=(10.0, 20.0, 40.0, 60.0),
    )


def frame(frame_id: int, timestamp_seconds: float | None = None) -> FrameState:
    return FrameState(
        frame_id=frame_id,
        timestamp_seconds=(
            float(frame_id) if timestamp_seconds is None else timestamp_seconds
        ),
    )


def test_fire_rule_emits_suspected_then_one_confirmed_event():
    rule = ConsecutiveFramesRule("fire_sequence", required_frames=3)

    first = rule.evaluate([fire_detection(0)], frame(0))
    second = rule.evaluate([fire_detection(1)], frame(1))
    third = rule.evaluate([fire_detection(2)], frame(2))
    repeated = rule.evaluate([fire_detection(3)], frame(3))

    assert [(event.event_type, event.alert_status) for event in first] == [
        ("fire_suspected", "suspected")
    ]
    assert first[0].track_id is None
    assert second == []
    assert [(event.event_type, event.alert_status) for event in third] == [
        ("fire_confirmed", "confirmed")
    ]
    assert repeated == []


def test_fire_rule_keeps_progress_across_one_allowed_gap():
    rule = ConsecutiveFramesRule(
        "fire_sequence",
        required_frames=3,
        max_gap_frames=1,
    )

    rule.evaluate([fire_detection(0)], frame(0))
    assert rule.evaluate([], frame(1)) == []
    assert rule.evaluate([fire_detection(2)], frame(2)) == []
    confirmed = rule.evaluate([fire_detection(3)], frame(3))

    assert len(confirmed) == 1
    assert confirmed[0].alert_status == "confirmed"


def test_fire_rule_resets_after_too_many_missing_frames():
    rule = ConsecutiveFramesRule(
        "fire_sequence",
        required_frames=3,
        max_gap_frames=1,
    )

    first = rule.evaluate([fire_detection(0)], frame(0))
    rule.evaluate([], frame(1))
    rule.evaluate([], frame(2))
    restarted = rule.evaluate([fire_detection(3)], frame(3))

    assert first[0].alert_status == "suspected"
    assert [(event.event_type, event.alert_status) for event in restarted] == [
        ("fire_suspected", "suspected")
    ]


@pytest.mark.parametrize(
    ("required_frames", "max_gap_frames"),
    [(0, 0), (1, -1)],
)
def test_fire_rule_rejects_invalid_frame_thresholds(
    required_frames: int,
    max_gap_frames: int,
):
    with pytest.raises(ValueError):
        ConsecutiveFramesRule(
            "fire_sequence",
            required_frames=required_frames,
            max_gap_frames=max_gap_frames,
        )


def test_rule_engine_applies_cooldown_between_fire_episodes():
    scene = SceneSpec.model_validate(
        {
            "scene_type": "fire",
            "model_id": "fire_smoke",
            "targets": ["fire", "smoke"],
            "rules": [
                {
                    "type": "consecutive_frames",
                    "params": {
                        "rule_id": "fire_sequence",
                        "frames": 2,
                        "max_gap_frames": 0,
                    },
                }
            ],
            "alert": {"enabled": True, "cooldown_seconds": 10},
        }
    )
    engine = RuleEngine()

    suspected = engine.evaluate(scene, [fire_detection(0)], frame(0))
    confirmed = engine.evaluate(scene, [fire_detection(1)], frame(1))
    engine.evaluate(scene, [], frame(2))
    repeated_suspected = engine.evaluate(scene, [fire_detection(3)], frame(3))
    repeated_confirmed = engine.evaluate(scene, [fire_detection(4)], frame(4))
    engine.evaluate(scene, [], frame(5))
    after_cooldown_suspected = engine.evaluate(
        scene,
        [fire_detection(12)],
        frame(12),
    )
    after_cooldown_confirmed = engine.evaluate(
        scene,
        [fire_detection(13)],
        frame(13),
    )

    assert [event.alert_status for event in suspected] == ["suspected"]
    assert [event.alert_status for event in confirmed] == ["confirmed"]
    assert repeated_suspected == []
    assert repeated_confirmed == []
    assert [event.alert_status for event in after_cooldown_suspected] == [
        "suspected"
    ]
    assert [event.alert_status for event in after_cooldown_confirmed] == [
        "confirmed"
    ]

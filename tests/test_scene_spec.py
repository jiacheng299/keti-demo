import pytest

from src.schemas.detection import Detection
from src.schemas.event import Event
from src.schemas.scene_spec import SceneSpec


def test_scene_spec_accepts_a_supported_border_configuration():
    scene = SceneSpec.model_validate(
        {
            "scene_type": "border",
            "model_id": "yolo_general",
            "targets": ["person"],
            "rules": [
                {
                    "type": "enter_region",
                    "params": {
                        "region_id": "restricted_zone",
                        "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                    },
                }
            ],
            "alert": {"enabled": True, "cooldown_seconds": 5},
        }
    )

    assert scene.scene_type == "border"
    assert scene.model_id == "yolo_general"
    assert scene.targets == ["person"]
    assert scene.rules[0].type == "enter_region"
    assert scene.alert.cooldown_seconds == 5


def test_scene_spec_rejects_removed_cross_line_rule():
    payload = {
        "scene_type": "border",
        "model_id": "yolo_general",
        "targets": ["person"],
        "rules": [{"type": "cross_line", "params": {}}],
    }

    with pytest.raises(ValueError, match="cross_line"):
        SceneSpec.model_validate(payload)


def test_scene_spec_rejects_an_unsupported_rule():
    payload = {
        "scene_type": "border",
        "model_id": "yolo_general",
        "targets": ["person"],
        "rules": [{"type": "run_shell", "params": {}}],
    }

    with pytest.raises(ValueError, match="run_shell"):
        SceneSpec.model_validate(payload)


def test_detection_rejects_an_inverted_bounding_box():
    with pytest.raises(ValueError, match="bbox_xyxy"):
        Detection(
            frame_id=12,
            class_name="person",
            confidence=0.9,
            bbox_xyxy=(50, 20, 10, 80),
        )


def test_event_accepts_the_shared_export_fields():
    event = Event(
        event_id="evt-001",
        scene_type="fire",
        event_type="fire_confirmed",
        target_class="fire",
        timestamp_seconds=3.2,
        confidence=0.88,
        trigger_rule="consecutive_frames",
        alert_status="confirmed",
        model_id="fire_smoke",
    )

    assert event.event_id == "evt-001"
    assert event.alert_status == "confirmed"
    assert event.track_id is None
    assert event.snapshot_path is None

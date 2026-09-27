import numpy as np

from src.rules.base_rule import EventCandidate
from src.schemas.detection import Detection
from src.schemas.scene_spec import SceneSpec
from src.visualization.scene_overlay import ALERT_COLOR, REGION_COLOR, SceneOverlay


def scene():
    return SceneSpec.model_validate({
        "scene_type": "border", "model_id": "yolo_general", "targets": ["person"],
        "rules": [{"type": "enter_region", "params": {
            "region_id": "zone", "polygon": [[160, 120], [480, 120], [480, 360], [160, 360]],
        }}],
    })


def candidate(event_type="enter_region", timestamp=1.0, track_id=7, status="confirmed"):
    return EventCandidate(
        frame_id=10, timestamp_seconds=timestamp, event_type=event_type,
        target_class="person" if track_id is not None else "fire", track_id=track_id,
        confidence=0.9, trigger_rule="zone", alert_status=status,
    )


def test_region_matches_configured_coordinates_without_changing_input():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output = SceneOverlay(scene()).draw(frame, [])
    assert np.count_nonzero(frame) == 0
    assert tuple(output[240, 160]) == REGION_COLOR
    assert np.any(output[240, 320])  # translucent region interior
    assert not np.any(output[400, 500])  # outside remains unchanged


def test_region_and_only_event_target_are_highlighted_until_video_time_expires():
    overlay = SceneOverlay(scene())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        Detection(frame_id=0, class_name="person", confidence=0.9, bbox_xyxy=(20, 300, 80, 360), track_id=7),
        Detection(frame_id=0, class_name="person", confidence=0.9, bbox_xyxy=(90, 300, 150, 360), track_id=8),
    ]
    overlay.update(1.0, [candidate()])
    overlay.update(3.99, [])
    active = overlay.draw(frame, detections)
    assert tuple(active[240, 160]) == ALERT_COLOR
    assert tuple(active[330, 20]) == ALERT_COLOR
    assert tuple(active[330, 90]) == (0, 220, 0)
    overlay.update(4.0, [])
    assert overlay.active_events == ()
    expired = overlay.draw(frame, detections)
    assert tuple(expired[240, 160]) == REGION_COLOR
    assert tuple(expired[330, 20]) == (0, 220, 0)
    assert not np.any(expired[20, 300])  # expired event banner is gone


def test_confirmation_replaces_suspected_notice_without_hiding_other_events():
    overlay = SceneOverlay(scene())
    overlay.update(1.0, [candidate("fire_suspected", track_id=None, status="suspected"), candidate()])
    overlay.update(1.3, [candidate("fire_confirmed", timestamp=1.3, track_id=None)])
    assert {event.event_type for event in overlay.active_events} == {"enter_region", "fire_confirmed"}


def test_multiple_rules_share_one_drawn_region():
    spec = scene()
    spec.rules.append(spec.rules[0].model_copy(update={"type": "leave_region"}))
    overlay = SceneOverlay(spec)
    assert len(overlay.regions) == 1


def test_overlay_handles_tiny_frames_and_long_identifiers():
    spec = scene()
    spec.rules[0].params["region_id"] = "long_zone_" * 50
    overlay = SceneOverlay(spec)
    overlay.update(1.0, [candidate()])
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    assert overlay.draw(frame, []).shape == frame.shape

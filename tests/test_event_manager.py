from typing import Literal

import cv2
import numpy as np
import pytest

from src.events.event_manager import EventManager
from src.rules.base_rule import EventCandidate


AlertStatus = Literal["suspected", "confirmed"]


def make_candidate(
    *,
    frame_id: int = 7,
    event_type: str = "fire_suspected",
    alert_status: AlertStatus = "suspected",
    track_id: int | None = None,
) -> EventCandidate:
    return EventCandidate(
        frame_id=frame_id,
        timestamp_seconds=frame_id / 25,
        event_type=event_type,
        target_class="fire" if track_id is None else "person",
        track_id=track_id,
        confidence=0.91,
        trigger_rule="fire_sequence" if track_id is None else "restricted_zone",
        alert_status=alert_status,
    )


def test_event_manager_builds_events_and_deduplicates(tmp_path):
    manager = EventManager(tmp_path, "fire", "fire_smoke")
    candidate = make_candidate()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    event = manager.add(candidate, frame)
    duplicate = manager.add(candidate, frame)

    assert event is not None
    assert event.event_id == "evt-000001"
    assert event.scene_type == "fire"
    assert event.model_id == "fire_smoke"
    assert event.snapshot_path is None
    assert duplicate is None


def test_event_manager_numbers_distinct_events_in_order(tmp_path):
    manager = EventManager(tmp_path, "fire", "fire_smoke")
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    first = manager.add(make_candidate(frame_id=7), frame)
    second = manager.add(make_candidate(frame_id=8), frame)

    assert first is not None
    assert second is not None
    assert first.event_id == "evt-000001"
    assert second.event_id == "evt-000002"


def test_confirmed_event_saves_readable_snapshot_and_track_id(tmp_path):
    manager = EventManager(tmp_path, "border", "yolo_general")
    candidate = make_candidate(
        frame_id=12,
        event_type="enter_region",
        alert_status="confirmed",
        track_id=4,
    )
    frame = np.full((32, 32, 3), 127, dtype=np.uint8)

    event = manager.add(candidate, frame)

    assert event is not None
    assert event.track_id == 4
    assert event.snapshot_path == "snapshots/evt-000001.jpg"
    encoded = np.fromfile(tmp_path / event.snapshot_path, dtype=np.uint8)
    assert cv2.imdecode(encoded, cv2.IMREAD_COLOR) is not None


def test_confirmed_event_rejects_invalid_frame_without_consuming_id(tmp_path):
    manager = EventManager(tmp_path, "fire", "fire_smoke")
    candidate = make_candidate(alert_status="confirmed")

    with pytest.raises(ValueError, match="BGR"):
        manager.add(candidate, np.zeros((32, 32), dtype=np.uint8))

    event = manager.add(candidate, np.zeros((32, 32, 3), dtype=np.uint8))

    assert event is not None
    assert event.event_id == "evt-000001"


def test_suspected_event_does_not_create_snapshot_directory(tmp_path):
    manager = EventManager(tmp_path, "fire", "fire_smoke")

    manager.add(
        make_candidate(alert_status="suspected"),
        np.zeros((32, 32, 3), dtype=np.uint8),
    )

    assert not (tmp_path / "snapshots").exists()


def test_snapshot_encode_failure_does_not_consume_event_id(tmp_path, monkeypatch):
    manager = EventManager(tmp_path, "fire", "fire_smoke")
    candidate = make_candidate(alert_status="confirmed")
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    with monkeypatch.context() as context:
        context.setattr(cv2, "imencode", lambda *_args, **_kwargs: (False, None))
        with pytest.raises(OSError, match="snapshot"):
            manager.add(candidate, frame)

    event = manager.add(candidate, frame)

    assert event is not None
    assert event.event_id == "evt-000001"

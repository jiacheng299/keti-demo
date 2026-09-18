from src.schemas.detection import Detection
from src.tracking.tracker import Tracker


def make_detection(frame_id: int, class_name: str, bbox) -> Detection:
    return Detection(
        frame_id=frame_id,
        class_name=class_name,
        confidence=0.9,
        bbox_xyxy=bbox,
    )


def test_tracker_keeps_an_id_for_a_nearby_same_class_detection():
    tracker = Tracker(max_distance=30.0, max_age_seconds=1.0)

    first = tracker.update(
        [make_detection(0, "person", (10, 10, 30, 50))],
        timestamp_seconds=0.0,
    )
    second = tracker.update(
        [make_detection(1, "person", (14, 11, 34, 51))],
        timestamp_seconds=0.1,
    )

    assert first[0].track_id == 0
    assert second[0].track_id == 0


def test_tracker_assigns_a_new_id_to_a_far_detection():
    tracker = Tracker(max_distance=20.0, max_age_seconds=1.0)

    first = tracker.update(
        [make_detection(0, "person", (0, 0, 20, 20))],
        timestamp_seconds=0.0,
    )
    second = tracker.update(
        [make_detection(1, "person", (80, 80, 100, 100))],
        timestamp_seconds=0.1,
    )

    assert first[0].track_id == 0
    assert second[0].track_id == 1

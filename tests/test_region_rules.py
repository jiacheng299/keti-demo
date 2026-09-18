from src.rules.base_rule import FrameState
from src.rules.region_rules import EnterRegionRule, LeaveRegionRule
from src.schemas.detection import Detection


SQUARE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def detection(frame_id: int, center: tuple[float, float]) -> Detection:
    x, y = center
    return Detection(
        frame_id=frame_id,
        class_name="person",
        confidence=0.9,
        bbox_xyxy=(x - 5, y - 5, x + 5, y + 5),
        track_id=7,
    )


def test_enter_region_fires_when_a_track_moves_from_outside_to_inside():
    rule = EnterRegionRule("restricted", SQUARE)

    assert rule.evaluate(
        [detection(0, (-10, 50))],
        FrameState(frame_id=0, timestamp_seconds=0.0),
    ) == []
    events = rule.evaluate(
        [detection(1, (20, 50))],
        FrameState(frame_id=1, timestamp_seconds=0.1),
    )

    assert len(events) == 1
    assert events[0].event_type == "enter_region"
    assert events[0].track_id == 7
    assert events[0].trigger_rule == "restricted"


def test_region_boundary_counts_as_inside():
    rule = EnterRegionRule("restricted", SQUARE)

    rule.evaluate(
        [detection(0, (-10, 50))],
        FrameState(frame_id=0, timestamp_seconds=0.0),
    )
    events = rule.evaluate(
        [detection(1, (0, 50))],
        FrameState(frame_id=1, timestamp_seconds=0.1),
    )

    assert len(events) == 1


def test_leave_region_fires_when_a_track_moves_from_inside_to_outside():
    rule = LeaveRegionRule("restricted", SQUARE)

    assert rule.evaluate(
        [detection(0, (50, 50))],
        FrameState(frame_id=0, timestamp_seconds=0.0),
    ) == []
    events = rule.evaluate(
        [detection(1, (110, 50))],
        FrameState(frame_id=1, timestamp_seconds=0.1),
    )

    assert len(events) == 1
    assert events[0].event_type == "leave_region"
    assert events[0].track_id == 7

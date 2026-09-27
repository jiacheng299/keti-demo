from types import SimpleNamespace

import numpy as np
import pytest

from src.llm.scene_parser import SceneParser
from src.models.face_landmarker_adapter import FaceLandmarkerAdapter, LEFT_EYE, RIGHT_EYE
from src.models.model_registry import ModelRegistry
from src.rules.base_rule import FrameState
from src.rules.drowsiness_rules import EyesClosedRule
from src.schemas.face_observation import FaceObservation
from src.tracking.face_tracker import FaceTracker
from src.visualization.scene_overlay import SceneOverlay, ALERT_COLOR


def face(track_id=1, ear=.1, box=(20, 100, 220, 350), **kwargs):
    return FaceObservation(valid=True, track_id=track_id, bbox_xyxy=box,
                           left_ear=ear, right_ear=ear, **kwargs)


def state(t, *faces):
    return FrameState(frame_id=round(t * 10), timestamp_seconds=t, effective_fps=10, faces=faces)


def test_reordered_detections_preserve_ids_but_disappearance_and_overlap_do_not():
    tracker = FaceTracker()
    left, right = face(box=(0, 0, 100, 100)), face(box=(200, 0, 300, 100))
    a, b = tracker.update([left, right], 0)
    assert a.track_id != b.track_id
    moved = left.model_copy(update={"bbox_xyxy": (3, 0, 103, 100)})
    new = tracker.update([right, moved], .1)
    assert [f.track_id for f in new] == [b.track_id, a.track_id]
    assert all(f.continuous for f in new)
    tracker.update([right], .2)
    new = tracker.update([left, right], .3)
    assert new[0].track_id not in {a.track_id, b.track_id}
    assert new[1].track_id == b.track_id
    tracker.update([], .4)
    overlap = [face(box=(0, 0, 100, 100)), face(box=(30, 0, 130, 100))]
    old = tracker.update(overlap, .5)
    new = tracker.update(overlap[::-1], .6)
    assert not any(f.continuous for f in new)
    assert not {f.track_id for f in old} & {f.track_id for f in new}
    ids = {f.track_id for f in new}
    assert not ids & {f.track_id for f in tracker.update(overlap, 2)}


def test_two_people_close_and_recover_independently_even_if_order_changes():
    rule = EyesClosedRule("eyes", seconds=1, recovery_seconds=.3)
    events = []
    for i in range(31):
        a = face(1, .1 if i < 15 else .3)
        b = face(2, .3 if i < 5 or i >= 25 else .1)
        faces = (a, b) if i % 2 else (b, a)
        events += rule.evaluate([], state(i / 10, *faces))
    assert [(e.track_id, e.event_type) for e in events] == [
        (1, "drowsiness_suspected"), (2, "drowsiness_suspected"),
        (1, "eyes_reopened"), (2, "eyes_reopened")]
    assert [e.timestamp_seconds for e in events] == pytest.approx([1, 1.5, 1.8, 2.8])
    assert {p.track_id for p in rule.progress} == {1, 2}


@pytest.mark.parametrize("interruption", ["missing", "invalid", "new_track", "discontinuous"])
def test_one_person_losing_evidence_does_not_interrupt_another(interruption):
    rule = EyesClosedRule("eyes", seconds=1)
    events = []
    for i in range(16):
        a = face(1)
        if i == 6 and interruption == "invalid":
            a = FaceObservation(track_id=1, reason="眼睛遮挡")
        if i == 6 and interruption == "discontinuous":
            a = face(1, continuous=False)
        if i >= 6 and interruption == "new_track":
            a = face(3)
        faces = (face(2),) if i == 6 and interruption == "missing" else (a, face(2))
        events += rule.evaluate([], state(i / 10, *faces))
    assert [(e.track_id, e.timestamp_seconds) for e in events] == [(2, 1)]
    assert len(rule._tracks) == 2
    rule.evaluate([], state(1.6))
    assert not rule._tracks
    assert rule.progress[0].status == "unknown"


def test_simultaneous_alarms_recovery_only_clears_matching_person_and_colors():
    scene = SceneParser(None, ModelRegistry.from_config("config/models.yaml")).load_template("drowsiness")
    overlay = SceneOverlay(scene)
    rule = EyesClosedRule("eyes", seconds=.2, recovery_seconds=.1)
    people = (face(1), face(2, box=(400, 100, 600, 350)))
    for i in range(3):
        overlay.update(i / 10, rule.evaluate([], state(i / 10, *people)))
    assert len(overlay.active_events) == 2
    for i in (3, 4):
        people = (face(1, .3), people[1])
        overlay.update(i / 10, rule.evaluate([], state(i / 10, *people)))
    assert {(e.track_id, e.event_type) for e in overlay.active_events} == {
        (1, "eyes_reopened"), (2, "drowsiness_suspected")}
    overlay.progress, overlay.face_observations = rule.progress, people
    image = overlay.draw(np.zeros((720, 900, 3), dtype=np.uint8), [])
    assert tuple(image[200, 20]) == (70, 190, 70)
    assert tuple(image[200, 400]) == ALERT_COLOR


def test_each_face_uses_its_own_landmarks_and_pose_matrix():
    points = [SimpleNamespace(x=.5, y=.5) for _ in range(478)]
    points[0] = SimpleNamespace(x=.1, y=.1)
    points[1] = SimpleNamespace(x=.9, y=.9)
    for indices, offset in ((LEFT_EYE, .2), (RIGHT_EYE, .6)):
        for index, (x, y) in zip(indices, [(0, .4), (.025, .39), (.075, .39), (.1, .4), (.075, .41), (.025, .41)]):
            points[index] = SimpleNamespace(x=x + offset, y=y)
    yaw = np.deg2rad(40)
    turned = np.eye(4)
    turned[:3, :3] = [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    matrices = [np.eye(4), turned]
    a = FaceLandmarkerAdapter._measure(points, matrices, 0, (500, 500, 3))
    b = FaceLandmarkerAdapter._measure(points, matrices, 1, (500, 500, 3))
    assert a.valid and b.valid
    assert a.left_ear == pytest.approx(.2)
    assert a.yaw_degrees == 0 and b.yaw_degrees == pytest.approx(40)
    assert not FaceLandmarkerAdapter._measure(points, matrices[:1], 1, (500, 500, 3)).valid


def test_duplicate_or_unassigned_multiface_ids_cannot_mix_timers():
    rule = EyesClosedRule("eyes")
    for faces in [(face(1), face(1)), (face(None), face(2))]:
        with pytest.raises(ValueError, match="track_id"):
            rule.evaluate([], state(0, *faces))

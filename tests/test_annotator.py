import numpy as np
import pytest

from src.schemas.detection import Detection
from src.visualization.annotator import Annotator


@pytest.mark.parametrize(
    ("track_id", "expected"),
    [(3, "person ID:3"), (None, "person")],
)
def test_annotator_label_contains_only_class_and_optional_id(track_id, expected):
    detection = Detection(
        frame_id=0,
        class_name="person",
        confidence=0.91,
        bbox_xyxy=(20, 20, 80, 90),
        track_id=track_id,
    )

    assert Annotator.format_label(detection) == expected


def test_annotator_draws_a_labeled_box_without_changing_the_input_frame():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    detection = Detection(
        frame_id=0,
        class_name="person",
        confidence=0.91,
        bbox_xyxy=(20, 20, 80, 90),
        track_id=3,
    )

    annotated = Annotator.draw(frame, [detection], scene_state={})

    assert annotated is not frame
    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(annotated) > 0
    assert annotated.shape == frame.shape

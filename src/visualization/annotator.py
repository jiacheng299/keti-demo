"""Draw normalized detections without mutating the source frame."""

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from src.schemas.detection import Detection


class Annotator:
    """Render boxes, class labels, and optional track IDs."""

    @staticmethod
    def draw(
        frame: np.ndarray,
        detections: Sequence[Detection],
        scene_state: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with three channels")
        _ = scene_state
        annotated = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = (int(value) for value in detection.bbox_xyxy)
            color = (0, 220, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = Annotator.format_label(detection)
            text_y = max(15, y1 - 6)
            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return annotated

    @staticmethod
    def format_label(detection: Detection) -> str:
        label = detection.class_name
        if detection.track_id is not None:
            label += f" ID:{detection.track_id}"
        return label

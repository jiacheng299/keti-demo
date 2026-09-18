"""CPU-only Ultralytics adapter that returns normalized detections."""

from typing import Any

import numpy as np

from src.schemas.detection import Detection

from .model_registry import ModelDefinition


class YoloAdapter:
    """Load one configured YOLO model and normalize its prediction output."""

    def __init__(self, definition: ModelDefinition, model: Any | None = None) -> None:
        self.definition = definition
        self._model = model

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.definition.weights.is_file():
            raise FileNotFoundError(f"model weights not found: {self.definition.weights}")

        from ultralytics import YOLO

        self._model = YOLO(str(self.definition.weights))

    def predict(self, frame: np.ndarray, frame_id: int = 0) -> list[Detection]:
        if frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with three channels")
        self.load()

        class_ids = [
            int(class_id)
            for class_id, class_name in self._model.names.items()
            if class_name in self.definition.classes
        ]
        results = self._model.predict(
            source=frame,
            imgsz=self.definition.imgsz,
            conf=self.definition.confidence,
            classes=class_ids,
            device=self.definition.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = self._as_numpy(result.boxes.xyxy)
            confidences = self._as_numpy(result.boxes.conf)
            detected_class_ids = self._as_numpy(result.boxes.cls)
            for box, confidence, class_id in zip(
                boxes, confidences, detected_class_ids, strict=True
            ):
                class_name = result.names[int(class_id)]
                confidence_value = float(confidence)
                if (
                    class_name not in self.definition.classes
                    or confidence_value < self.definition.confidence
                ):
                    continue
                detections.append(
                    Detection(
                        frame_id=frame_id,
                        class_name=class_name,
                        confidence=confidence_value,
                        bbox_xyxy=tuple(float(value) for value in box),
                    )
                )
        return detections

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def unload(self) -> None:
        self._model = None

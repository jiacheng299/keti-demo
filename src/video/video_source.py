"""Validated OpenCV source for local video files."""

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Iterator, Self

import cv2
import numpy as np


class VideoOpenError(ValueError):
    """Raised when a local file cannot be opened as a usable video."""


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


class VideoSource:
    """Own one video capture and expose normalized frames and timestamps."""

    def __init__(
        self,
        path: Path,
        capture: cv2.VideoCapture,
        metadata: VideoMetadata,
    ) -> None:
        self.path = path
        self._capture = capture
        self._metadata = metadata

    @classmethod
    def open(cls, path: str | Path) -> "VideoSource":
        video_path = Path(path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise VideoOpenError(f"cannot open video: {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0 or frame_count < 0:
            capture.release()
            raise VideoOpenError(f"video has invalid metadata: {video_path}")

        metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=frame_count / fps,
        )
        return cls(video_path, capture, metadata)

    @property
    def is_open(self) -> bool:
        return self._capture.isOpened()

    def metadata(self) -> VideoMetadata:
        return self._metadata

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        if not self.is_open:
            raise RuntimeError("video source is closed")

        frame_id = 0
        while True:
            readable, frame = self._capture.read()
            if not readable:
                break
            yield frame_id, frame_id / self._metadata.fps, frame
            frame_id += 1

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> Self:
        if not self.is_open:
            raise RuntimeError("video source is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

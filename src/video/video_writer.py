"""Validated OpenCV writer for annotated result videos."""

from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np


class VideoWriteError(ValueError):
    """Raised when an output video cannot be created or receives bad frames."""


class VideoWriter:
    """Own one OpenCV writer and enforce a fixed output frame size."""

    def __init__(
        self,
        path: Path,
        writer: cv2.VideoWriter,
        frame_size: tuple[int, int],
    ) -> None:
        self.path = path
        self._writer = writer
        self._frame_size = frame_size

    @classmethod
    def open(
        cls,
        path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ) -> "VideoWriter":
        if fps <= 0:
            raise VideoWriteError("fps must be greater than zero")
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise VideoWriteError("frame_size values must be greater than zero")
        if len(codec) != 4:
            raise VideoWriteError("codec must contain exactly four characters")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if not writer.isOpened():
            writer.release()
            raise VideoWriteError(f"cannot create output video: {output_path}")
        return cls(output_path, writer, frame_size)

    @property
    def is_open(self) -> bool:
        return self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        if not self.is_open:
            raise VideoWriteError("video writer is closed")
        width, height = self._frame_size
        if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
            raise VideoWriteError(
                f"frame must have shape {(height, width, 3)} and dtype uint8"
            )
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> Self:
        if not self.is_open:
            raise VideoWriteError("video writer is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

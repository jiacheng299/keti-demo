"""Stable pipeline results consumed by progress displays and the UI."""

from dataclasses import dataclass
from pathlib import Path

from src.schemas.event import Event


@dataclass(frozen=True)
class PipelineProgress:
    frames_read: int
    total_frames: int
    processed_frames: int
    event_count: int

    @property
    def fraction(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(1.0, max(0.0, self.frames_read / self.total_frames))


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    result_video_path: Path
    events: tuple[Event, ...]
    frames_read: int
    processed_frames: int
    detection_count: int
    stopped: bool
    elapsed_seconds: float

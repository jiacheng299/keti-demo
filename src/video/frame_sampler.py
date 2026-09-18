"""Deterministic frame sampling for CPU-limited processing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSampler:
    """Process frame zero and then every configured number of frames."""

    every_n_frames: int = 1

    def __post_init__(self) -> None:
        if self.every_n_frames <= 0:
            raise ValueError("every_n_frames must be greater than zero")

    def should_process(self, frame_id: int) -> bool:
        if frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        return frame_id % self.every_n_frames == 0

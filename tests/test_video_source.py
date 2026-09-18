from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video.video_source import VideoOpenError, VideoSource
from src.video.video_writer import VideoWriter


FRAME_SIZE = (64, 48)
FPS = 10.0


def create_synthetic_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        FPS,
        FRAME_SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError("test environment cannot create an MJPG video")
    try:
        for index in range(frame_count):
            frame = np.full(
                (FRAME_SIZE[1], FRAME_SIZE[0], 3),
                fill_value=index * 40,
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()


def test_video_source_reads_metadata_frame_ids_and_timestamps(tmp_path):
    video_path = tmp_path / "input.avi"
    create_synthetic_video(video_path)

    with VideoSource.open(video_path) as source:
        metadata = source.metadata()
        frames = list(source)

    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.fps == pytest.approx(10.0, rel=0.05)
    assert metadata.frame_count == 3
    assert metadata.duration_seconds == pytest.approx(0.3, rel=0.05)
    assert [frame_id for frame_id, _, _ in frames] == [0, 1, 2]
    assert [timestamp for _, timestamp, _ in frames] == pytest.approx(
        [0.0, 0.1, 0.2], rel=0.05
    )
    assert all(frame.shape == (48, 64, 3) for _, _, frame in frames)
    assert not source.is_open


def test_video_source_rejects_an_unreadable_file(tmp_path):
    broken_path = tmp_path / "broken.avi"
    broken_path.write_bytes(b"not a video")

    with pytest.raises(VideoOpenError, match="broken.avi"):
        VideoSource.open(broken_path)


def test_video_source_closes_after_an_exception(tmp_path):
    video_path = tmp_path / "input.avi"
    create_synthetic_video(video_path)
    source = VideoSource.open(video_path)

    with pytest.raises(RuntimeError, match="processing failed"):
        with source:
            raise RuntimeError("processing failed")

    assert not source.is_open


def test_video_writer_creates_a_readable_video_and_closes(tmp_path):
    output_path = tmp_path / "output.avi"

    with VideoWriter.open(
        output_path,
        fps=FPS,
        frame_size=FRAME_SIZE,
        codec="MJPG",
    ) as writer:
        for index in range(3):
            frame = np.full(
                (FRAME_SIZE[1], FRAME_SIZE[0], 3),
                fill_value=index * 60,
                dtype=np.uint8,
            )
            writer.write(frame)

    capture = cv2.VideoCapture(str(output_path))
    try:
        readable_frames = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            readable_frames += 1
    finally:
        capture.release()

    assert readable_frames == 3
    assert not writer.is_open

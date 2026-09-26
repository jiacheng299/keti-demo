from pathlib import Path

import cv2
import numpy as np

from src.video.browser_video import make_browser_playable


def create_mp4v_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("test environment cannot create an MP4V video")
    try:
        for index in range(frame_count):
            writer.write(np.full((48, 64, 3), index * 40, dtype=np.uint8))
    finally:
        writer.release()


def count_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def test_make_browser_playable_transcodes_to_h264_without_losing_frames(tmp_path):
    source = tmp_path / "source.mp4"
    destination = tmp_path / "browser.mp4"
    create_mp4v_video(source)

    output = make_browser_playable(source, destination)

    assert output == destination
    assert output.is_file()
    assert b"avc1" in output.read_bytes()
    assert count_frames(output) == 3

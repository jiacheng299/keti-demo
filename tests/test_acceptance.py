import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.validation.acceptance import (
    AcceptanceValidationError,
    validate_run_artifacts,
)
from src.video.browser_video import make_browser_playable


def create_video(path: Path, *, browser_playable: bool = True) -> None:
    source = path if not browser_playable else path.with_name("source.mp4")
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("test environment cannot create video")
    try:
        for value in [0, 40, 80]:
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    if browser_playable:
        make_browser_playable(source, path)


def create_complete_run(run_dir: Path) -> None:
    run_dir.mkdir()
    event = {
        "event_id": "evt-000001",
        "scene_type": "border",
        "event_type": "enter_region",
        "target_class": "person",
        "track_id": 1,
        "timestamp_seconds": 1.1,
        "confidence": 0.9,
        "trigger_rule": "restricted_zone",
        "snapshot_path": "snapshots/evt-000001.jpg",
        "alert_status": "confirmed",
        "model_id": "yolo_general",
    }
    (run_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "events.json").write_text(
        json.dumps([event], ensure_ascii=False),
        encoding="utf-8",
    )
    with (run_dir / "events.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(event))
        writer.writeheader()
        writer.writerow(event)
    (run_dir / "summary.json").write_text(
        json.dumps({"event_count": 1, "metrics": {"frames_read": 3}}),
        encoding="utf-8",
    )
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir()
    encoded, jpeg = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert encoded
    (snapshot_dir / "evt-000001.jpg").write_bytes(jpeg.tobytes())
    create_video(run_dir / "result.mp4")


def test_validate_run_artifacts_checks_cross_file_counts_video_and_snapshot(tmp_path):
    run_dir = tmp_path / "run"
    create_complete_run(run_dir)

    evidence = validate_run_artifacts(run_dir, expected_min_events=1)

    assert evidence == {
        "event_count": 1,
        "csv_event_count": 1,
        "snapshot_count": 1,
        "video_frame_count": 3,
        "browser_codec": "h264",
    }


def test_validate_run_artifacts_rejects_missing_required_file(tmp_path):
    run_dir = tmp_path / "run"
    create_complete_run(run_dir)
    (run_dir / "events.csv").unlink()

    with pytest.raises(AcceptanceValidationError, match="missing artifact: events.csv"):
        validate_run_artifacts(run_dir, expected_min_events=1)


def test_validate_run_artifacts_rejects_event_count_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    create_complete_run(run_dir)
    (run_dir / "summary.json").write_text(
        json.dumps({"event_count": 2, "metrics": {"frames_read": 3}}),
        encoding="utf-8",
    )

    with pytest.raises(AcceptanceValidationError, match="event counts do not match"):
        validate_run_artifacts(run_dir, expected_min_events=1)


def test_validate_run_artifacts_rejects_unreadable_or_non_h264_video(tmp_path):
    run_dir = tmp_path / "run"
    create_complete_run(run_dir)
    create_video(run_dir / "result.mp4", browser_playable=False)

    with pytest.raises(AcceptanceValidationError, match="result video is not H.264"):
        validate_run_artifacts(run_dir, expected_min_events=1)

    (run_dir / "result.mp4").write_bytes(b"avc1-not-a-video")
    with pytest.raises(AcceptanceValidationError, match="result video is unreadable"):
        validate_run_artifacts(run_dir, expected_min_events=1)

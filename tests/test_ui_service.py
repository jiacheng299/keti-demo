import pytest

from src.ui.app_service import (
    UnsupportedVideoTypeError,
    resolve_scene_spec,
    save_uploaded_video,
    snapshot_paths_for_display,
)


class FailIfCalledClient:
    def complete(self, _messages):
        raise AssertionError("offline template mode must not call DeepSeek")


def test_offline_scene_resolution_does_not_call_deepseek():
    scene = resolve_scene_spec(
        mode="offline",
        requirement="",
        template_id="fire_detection",
        client=FailIfCalledClient(),
    )

    assert scene.scene_type == "fire"
    assert scene.model_id == "fire_smoke"


def test_uploaded_video_uses_content_hash_and_safe_extension(tmp_path):
    output = save_uploaded_video(
        b"video-bytes",
        "..\\camera.clip.ogv",
        tmp_path,
    )

    assert output.parent == tmp_path
    assert output.suffix == ".ogv"
    assert output.name != "camera.clip.ogv"
    assert output.read_bytes() == b"video-bytes"


def test_uploaded_video_rejects_unapproved_extension(tmp_path):
    with pytest.raises(UnsupportedVideoTypeError, match="unsupported video type"):
        save_uploaded_video(b"not-video", "payload.exe", tmp_path)


def test_snapshot_paths_only_return_existing_files_inside_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(parents=True)
    valid_snapshot = snapshot_dir / "evt-000001.jpg"
    valid_snapshot.write_bytes(b"jpeg")
    outside_snapshot = tmp_path / "outside.jpg"
    outside_snapshot.write_bytes(b"private")
    events = [
        {"event_id": "evt-000001", "snapshot_path": "snapshots/evt-000001.jpg"},
        {"event_id": "evt-000002", "snapshot_path": "../outside.jpg"},
        {"event_id": "evt-000003", "snapshot_path": "snapshots/missing.jpg"},
        {"event_id": "evt-000004", "snapshot_path": None},
    ]

    snapshots = snapshot_paths_for_display(run_dir, events)

    assert snapshots == [(valid_snapshot.resolve(), events[0])]

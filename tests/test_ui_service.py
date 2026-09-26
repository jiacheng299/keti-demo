import pytest

from src.ui.app_service import (
    UnsupportedVideoTypeError,
    resolve_scene_spec,
    save_uploaded_video,
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

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_upload_scene_controls_and_offline_status(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    app = AppTest.from_file(APP_PATH).run()

    assert not app.exception
    assert any("开放场景视觉语义认知 Demo" in item.value for item in app.title)
    assert app.file_uploader[0].label == "上传视频文件"
    assert app.text_area[0].label == "场景需求"
    assert app.selectbox[0].label == "离线模板"
    assert app.button(key="prepare_scene").label == "生成场景配置"
    assert app.button(key="run_analysis").disabled is True
    assert any("DeepSeek API：未配置" in item.value for item in app.warning)


def test_offline_template_can_be_generated_before_video_upload(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    app = AppTest.from_file(APP_PATH).run()

    app.button(key="prepare_scene").click().run()

    assert not app.exception
    assert app.session_state["scene_spec"]["model_id"] == "yolo_general"
    assert app.button(key="run_analysis").disabled is True
    assert app.json


def test_completed_run_renders_video_localized_events_and_downloads(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result_video = run_dir / "result.mp4"
    result_video.write_bytes(b"fake-mp4")
    for filename in ["events.json", "events.csv", "summary.json", "config.json"]:
        (run_dir / filename).write_text("[]", encoding="utf-8")

    app = AppTest.from_file(APP_PATH)
    app.session_state["run_result"] = {
        "run_dir": str(run_dir),
        "result_video_path": str(result_video),
        "frames_read": 12,
        "processed_frames": 6,
        "detection_count": 9,
        "event_count": 1,
        "elapsed_seconds": 2.5,
        "events": [
            {
                "event_id": "evt-000001",
                "scene_type": "border",
                "event_type": "enter_region",
                "target_class": "person",
                "track_id": 3,
                "timestamp_seconds": 1.2,
                "confidence": 0.91,
                "trigger_rule": "restricted_zone",
                "snapshot_path": "snapshots/evt-000001.jpg",
                "alert_status": "confirmed",
                "model_id": "yolo_general",
            }
        ],
        "model_id": "yolo_general",
    }
    app.run()

    assert not app.exception
    assert app.get("video")
    assert list(app.dataframe[0].value.columns) == [
        "事件编号",
        "事件类型",
        "目标类别",
        "目标 ID",
        "视频时间（秒）",
        "状态",
    ]
    assert len(app.get("download_button")) == 5

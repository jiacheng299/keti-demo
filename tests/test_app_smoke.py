from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_upload_scene_controls_and_offline_status(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    app = AppTest.from_file(APP_PATH).run()

    assert not app.exception
    assert any(item.value == "开放场景视觉语义认知" for item in app.title)
    assert app.file_uploader[0].label == "上传视频文件"
    assert len(app.text_area) == 0
    assert app.selectbox[0].label == "离线模板"
    assert app.selectbox[0].options == ["边防禁区", "边防人员滞留", "火灾检测", "人员在岗监测", "打瞌睡监测"]
    assert app.button(key="prepare_scene").label == "生成场景配置"
    assert app.button(key="run_analysis").disabled is True
    assert any("DeepSeek API：未配置" in item.value for item in app.warning)


def test_offline_template_can_be_generated_before_video_upload(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    app = AppTest.from_file(APP_PATH).run()

    app.button(key="prepare_scene").click().run()

    assert not app.exception
    assert app.session_state["scene_spec"]["model_id"] == "yolo_general"
    assert app.multiselect(key="cfg_targets").options == ["人员", "车辆"]
    assert app.button(key="run_analysis").disabled is True
    assert app.json


def test_scene_modes_only_show_relevant_inputs_and_invalidate_previous_config():
    app = AppTest.from_file(APP_PATH).run()
    app.button(key="prepare_scene").click().run()
    assert app.session_state["scene_spec"] is not None
    app.segmented_control(key="scene_mode").set_value("DeepSeek 解析").run()
    assert not app.exception
    assert len(app.selectbox) == 0
    assert app.text_area[0].label == "场景需求"
    assert app.session_state["scene_spec"] is None
    assert app.button(key="run_analysis").disabled
    app.segmented_control(key="scene_mode").set_value("离线模板").run()
    assert not app.exception
    assert len(app.text_area) == 0
    assert app.selectbox[0].label == "离线模板"
    app.button(key="prepare_scene").click().run()
    assert app.session_state["scene_spec"] is not None
    app.selectbox(key="template_label").select("火灾检测").run()
    assert app.session_state["scene_spec"] is None


def test_completed_run_renders_video_localized_events_and_downloads(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result_video = run_dir / "result.mp4"
    result_video.write_bytes(b"fake-mp4")
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir()
    encoded, jpeg = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert encoded
    (snapshot_dir / "evt-000001.jpg").write_bytes(jpeg.tobytes())
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
    assert {b.key for b in app.get("download_button") if b.key and b.key.startswith("download_")} == {
        "download_video", "download_events.json", "download_events.csv", "download_summary.json", "download_config.json",
    }
    assert len(app.image) == 1

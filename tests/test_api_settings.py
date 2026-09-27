import json
import os
from pathlib import Path
from urllib.error import HTTPError

import pytest
import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from src.llm.deepseek_client import DeepSeekAPIError, DeepSeekClient
from src.security.credential_store import load_api_key, save_api_key

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_page_updates_qwen_independently_and_remembers_it_between_sessions(monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    save_api_key("existing-deepseek")
    app = AppTest.from_file(APP).run()
    for value in ("qwen-original", "qwen-replacement"):
        app.text_input(key="qwen_key_input").set_value(value)
        app.button(key="save_qwen_api_key").click().run()
        assert not app.exception
        assert load_api_key("qwen") == value
        assert load_api_key() == "existing-deepseek"
        assert app.text_input(key="qwen_key_input").value == ""
    app.text_input(key="qwen_key_input").set_value("invalid key")
    app.button(key="save_qwen_api_key").click().run()
    assert load_api_key("qwen") == "qwen-replacement"
    assert app.text_input(key="qwen_key_input").value == ""
    assert any("不能包含空白" in message.value for message in app.error)
    new_session = AppTest.from_file(APP).run()
    assert any("Qwen API：已配置（系统凭据）" in message.value for message in new_session.success)
    assert new_session.text_input(key="qwen_key_input").value == ""
    new_session.button(key="clear_qwen_api_key").click().run()
    assert load_api_key("qwen") is None
    assert load_api_key() == "existing-deepseek"


def test_qwen_environment_fallback_is_explicit_after_deleting_saved_key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "environment-qwen-test")
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    save_api_key("stored-qwen-test", "qwen")
    app = AppTest.from_file(APP).run()
    assert any("Qwen API：已配置（系统凭据）" in message.value for message in app.success)
    app.button(key="clear_qwen_api_key").click().run()
    assert any("Qwen API：已配置（环境变量）" in message.value for message in app.success)
    assert app.text_input(key="qwen_key_input").value == ""


def assessed_scene(payload, text):
    return {"status": "supported", "reason": "支持火焰检测", "summary": text,
            "requirements": [{"source_text": text, "capability": "consecutive_frames", "rule_index": 0}],
            "unmet_requirements": [], "scene": payload}


def test_page_saves_key_clears_password_and_loads_it_in_a_new_session(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    app = AppTest.from_file(APP).run()
    app.text_input(key="deepseek_key_input").set_value("  page-test-key  ")
    app.button(key="save_api_key").click().run()
    assert not app.exception
    assert load_api_key() == "page-test-key"
    assert app.text_input(key="deepseek_key_input").value == ""
    assert "DEEPSEEK_API_KEY" not in os.environ
    new_session = AppTest.from_file(APP).run()
    assert any("系统凭据" in message.value for message in new_session.success)
    assert new_session.text_input(key="deepseek_key_input").value == ""
    new_session.button(key="clear_api_key").click().run()
    assert load_api_key() is None
    assert any("未配置" in message.value for message in new_session.warning)


def test_page_passes_saved_key_and_uses_actual_deepseek_configuration(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-test-key")
    save_api_key("page-test-key")
    calls = []
    payload = {
        "scene_type": "fire", "model_id": "fire_smoke", "targets": ["fire"],
        "rules": [{"type": "consecutive_frames", "params": {"frames": 15, "max_gap_frames": 0}}],
    }

    def transport(request, timeout):
        calls.append((request.headers["Authorization"], json.loads(request.data)))
        content = assessed_scene(payload, json.loads(request.data)["messages"][1]["content"])
        return json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()

    monkeypatch.setattr("src.llm.deepseek_client._urlopen_transport", transport)
    app = AppTest.from_file(APP).run()
    assert calls == []
    app.segmented_control(key="scene_mode").set_value("DeepSeek 解析").run()
    app.text_area(key="scene_requirement").set_value("连续15帧火焰才确认")
    app.button(key="prepare_scene").click().run()
    assert not app.exception
    assert calls[0][0] == "Bearer page-test-key"
    assert app.session_state["scene_spec"]["rules"][0]["params"]["frames"] == 15
    assert app.session_state["scene_source"] == "DeepSeek"
    assert app.session_state["scene_warning"] is None


@pytest.mark.parametrize("status,expected", [(401, "无效"), (402, "余额不足"), (429, "过于频繁")])
def test_page_explains_api_failure_without_using_a_hidden_template(monkeypatch, status, expected):
    save_api_key("page-test-key")

    def transport(request, timeout):
        raise HTTPError(request.full_url, status, "private details page-test-key", {}, None)

    monkeypatch.setattr("src.llm.deepseek_client._urlopen_transport", transport)
    app = AppTest.from_file(APP).run()
    app.segmented_control(key="scene_mode").set_value("DeepSeek 解析").run()
    app.text_area(key="scene_requirement").set_value("人员停留3秒")
    app.button(key="prepare_scene").click().run()
    assert not app.exception
    assert app.session_state["scene_spec"] is None
    message = app.error[0].value
    assert expected in message and "切换到“离线模板”" in message
    assert "page-test-key" not in message
    assert not app.json
    assert app.button(key="run_analysis").disabled


def test_missing_key_is_explained_without_a_network_call(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    def fail(*args):
        raise AssertionError("No network request should occur")

    monkeypatch.setattr("src.llm.deepseek_client._urlopen_transport", fail)
    app = AppTest.from_file(APP).run()
    app.segmented_control(key="scene_mode").set_value("DeepSeek 解析").run()
    app.text_area(key="scene_requirement").set_value("人员停留3秒")
    app.button(key="prepare_scene").click().run()
    assert not app.exception
    assert "未配置 API Key" in app.error[0].value
    assert app.session_state["scene_spec"] is None


@pytest.mark.parametrize("code,expected", [
    ("timeout", "超时"),
    ("connection_error", "无法连接"),
    ("empty_response", "空的最终答案"),
    ("output_truncated", "长度上限"),
    ("invalid_response", "不是有效的 DeepSeek 响应"),
])
def test_page_distinguishes_api_error_causes(monkeypatch, code, expected):
    save_api_key("page-test-key")

    def complete(*args):
        raise DeepSeekAPIError("private provider details", error_code=code)

    monkeypatch.setattr(DeepSeekClient, "complete", complete)
    app = AppTest.from_file(APP).run()
    app.segmented_control(key="scene_mode").set_value("DeepSeek 解析").run()
    app.text_area(key="scene_requirement").set_value("人员停留3秒")
    app.button(key="prepare_scene").click().run()
    assert not app.exception
    assert expected in app.error[0].value
    assert "private provider details" not in app.error[0].value
    assert app.session_state["scene_spec"] is None


def test_page_requires_regeneration_when_duration_video_fps_changes(monkeypatch, tmp_path):
    save_api_key("page-test-key")
    payload = {
        "scene_type": "fire", "model_id": "fire_smoke", "targets": ["fire"],
        "rules": [{"type": "consecutive_frames", "params": {"frames": 201, "max_gap_frames": 0}}],
    }
    messages_seen = []

    def complete(self, messages):
        messages_seen.append(messages)
        return json.dumps(assessed_scene(payload, messages[1]["content"]))

    monkeypatch.setattr(DeepSeekClient, "complete", complete)
    app = AppTest.from_file(APP).run()
    for fps in (25, 10):
        path = tmp_path / f"sample-{fps}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
        assert writer.isOpened()
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        app.file_uploader[0].set_value((path.name, path.read_bytes(), "video/x-msvideo")).run()
        if fps == 25:
            app.segmented_control(key="scene_mode").set_value("DeepSeek 解析").run()
            app.text_area(key="scene_requirement").set_value("连续监测到火焰8秒后报警")
            app.button(key="prepare_scene").click().run()
            assert not app.button(key="run_analysis").disabled
            assert any("201 个命中帧" in item.value and "8 秒" in item.value for item in app.caption)
        else:
            assert app.button(key="run_analysis").disabled
            assert any("帧率与生成配置时不同" in item.value for item in app.warning)
        assert not app.exception
    assert "25 FPS" in messages_seen[0][0]["content"]

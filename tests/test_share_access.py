from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_shared_instance_opens_directly_and_hides_credential_management(monkeypatch):
    monkeypatch.setenv("DEMO_SHARED_INSTANCE", "1")
    monkeypatch.setenv("DEMO_SHARE_CODE", "demo-test-code-123")
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert app.file_uploader
    assert not any(t.label == "访问码" for t in app.text_input)
    assert not any(b.key in {"save_api_key", "clear_api_key", "save_qwen_api_key", "clear_qwen_api_key"} for b in app.button)
    assert not any(t.label in {"DeepSeek API Key", "Qwen API Key"} for t in app.text_input)


def test_shared_instance_does_not_need_configured_code(monkeypatch):
    monkeypatch.setenv("DEMO_SHARED_INSTANCE", "1")
    monkeypatch.delenv("DEMO_SHARE_CODE", raising=False)
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert app.file_uploader

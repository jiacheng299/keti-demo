from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_renders_title_and_offline_status():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()

    assert not app.exception
    assert any("开放场景视觉语义认知 Demo" in item.value for item in app.title)
    assert any("当前阶段：项目骨架" in item.value for item in app.markdown)

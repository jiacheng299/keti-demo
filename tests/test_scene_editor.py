import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.llm.scene_parser import SceneParser
from src.models.model_registry import ModelRegistry
from src.schemas.scene_spec import SceneSpec
from src.schemas.scene_time import resolve_time_rules
from src.ui.scene_editor import validate_geometry


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("polygon",[
    [[0,0],[700,0],[700,20]],
    [[0,0],[0,0],[10,10]],
    [[0,0],[10,0],[20,0]],
    [[0,0],[100,100],[0,100],[80,0]],
])
def test_canvas_rejects_invalid_geometry(polygon):
    with pytest.raises(ValueError):
        validate_geometry(polygon,640,480)


def test_canvas_accepts_concave_polygon_in_original_pixels():
    validate_geometry([[30,30],[600,30],[300,200],[600,450],[30,450]],640,480)


def test_duration_is_calculated_locally_and_rebased_for_sampling():
    scene=SceneSpec(scene_type="fire",model_id="fire_smoke",targets=["fire"],rules=[{"type":"consecutive_frames","params":{"seconds":8,"frames":3}}])
    assert resolve_time_rules(scene,25).rules[0].params["frames"] == 201
    assert resolve_time_rules(scene,10).rules[0].params["frames"] == 81
    assert resolve_time_rules(scene,25,3).rules[0].params["frames"] == 68
    assert scene.rules[0].params["frames"] == 3


def test_parser_accepts_duration_intent_without_model_calculating_frames():
    class Client:
        def complete(self,messages):
            scene = {"scene_type":"fire","model_id":"fire_smoke","targets":["fire"],"rules":[{"type":"consecutive_frames","params":{"seconds":8,"max_gap_frames":0}}]}
            return json.dumps({"status":"supported","reason":"支持持续火焰检测","summary":"火焰8秒", "requirements":[{"source_text":"火焰8秒","capability":"consecutive_frames","rule_index":0}],"unmet_requirements":[],"scene":scene})
    scene=SceneParser(Client(),ModelRegistry.from_config(ROOT/"config/models.yaml")).parse("火焰8秒",video_fps=25)
    assert scene.rules[0].params["frames"] == 201
    assert scene.rules[0].params["seconds"] == 8


def upload(app,name):
    video=ROOT/"assets/demo_videos"/name
    if not video.exists():
        pytest.skip("downloaded demo video is not installed")
    app.file_uploader[0].set_value((video.name,video.read_bytes(),"video/x-msvideo")).run()


def test_edit_cards_apply_invalidate_and_add_remove_regions():
    app=AppTest.from_file(ROOT/"app.py").run()
    upload(app,"border_demo.avi")
    app.button(key="prepare_scene").click().run()
    assert not app.exception
    app.button(key="cfg_apply").click().run()
    assert not app.button(key="run_analysis").disabled
    app.number_input(key="cfg_cooldown").set_value(12.0).run()
    assert app.button(key="run_analysis").disabled
    assert app.session_state["scene_spec"]["alert"]["cooldown_seconds"] == 5
    app.button(key="cfg_apply").click().run()
    assert app.session_state["scene_spec"]["alert"]["cooldown_seconds"] == 12
    app.button(key="cfg_add").click().run()
    app.button(key="cfg_apply").click().run()
    assert len(app.session_state["scene_spec"]["rules"]) == 2
    app.button(key="cfg_delete").click().run()
    app.button(key="cfg_apply").click().run()
    assert len(app.session_state["scene_spec"]["rules"]) == 1
    assert not app.exception


def test_fire_cards_convert_seconds_and_require_apply_after_fps_change():
    app=AppTest.from_file(ROOT/"app.py").run()
    upload(app,"fire_demo.avi")
    app.selectbox(key="template_label").select("火灾检测").run()
    app.button(key="prepare_scene").click().run()
    app.radio[0].set_value("秒").run()
    assert app.button(key="run_analysis").disabled
    app.button(key="cfg_apply").click().run()
    assert app.session_state["scene_spec"]["rules"][0]["params"]["frames"] == 201
    assert not app.button(key="run_analysis").disabled
    upload(app,"border_demo.avi")
    assert app.button(key="run_analysis").disabled
    app.button(key="cfg_apply").click().run()
    assert app.session_state["scene_spec"]["rules"][0]["params"]["frames"] == 81
    assert not app.exception

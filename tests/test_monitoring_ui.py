from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("label,scene_type,video",[
    ("人员在岗监测","on_duty","on_duty_demo.mp4"),
    ("打瞌睡监测","drowsiness","drowsiness_demo.mp4"),
])
def test_new_template_cards_apply_and_invalidate(label,scene_type,video):
    app=AppTest.from_file(ROOT/'app.py').run()
    path=ROOT/'assets/demo_videos'/video
    if not path.is_file(): pytest.skip('local example video not installed')
    app.file_uploader[0].set_value((path.name,path.read_bytes(),'video/mp4')).run()
    app.selectbox(key='template_label').select(label).run()
    app.button(key='prepare_scene').click().run()
    assert not app.exception
    assert app.session_state['scene_spec']['scene_type']==scene_type
    assert app.multiselect(key='cfg_targets').options==['人员']
    assert app.multiselect(key='cfg_targets').disabled
    app.button(key='cfg_apply').click().run()
    assert not app.button(key='run_analysis').disabled
    duration=next(x for x in app.number_input if '报警（秒）' in x.label)
    duration.set_value(3.0).run()
    assert app.button(key='run_analysis').disabled
    app.button(key='cfg_apply').click().run()
    assert app.session_state['scene_spec']['rules'][0]['params']['seconds']==3
    if scene_type=='on_duty':
        app.button(key='cfg_add').click().run()
        app.button(key='cfg_apply').click().run()
        assert len(app.session_state['scene_spec']['rules'])==2
        assert all(r['type']=='region_understaffed' for r in app.session_state['scene_spec']['rules'])
    else:
        next(x for x in app.number_input if x.label=='闭眼 EAR 阈值').set_value(.3).run()
        assert app.button(key='cfg_apply').disabled
        assert app.button(key='run_analysis').disabled
    assert not app.exception

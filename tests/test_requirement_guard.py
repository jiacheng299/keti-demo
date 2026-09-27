import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.llm.requirement_guard import RequirementRejected, validate_requirement_text
from src.llm.scene_parser import SceneParser, SceneParseError
from src.models.model_registry import ModelRegistry
from src.ui.app_service import resolve_scene_with_status


ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        result = self.responses.pop(0)
        return json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else result


def decision(text, template="on_duty"):
    parser=SceneParser(None,ModelRegistry.from_config(ROOT/'config/models.yaml'),template_dir=ROOT/'config/scenes')
    scene=parser.load_template(template).model_dump(mode="json")
    return dict(status="supported",reason="支持现有规则",summary=text,
                requirements=[dict(source_text=text,capability=scene['rules'][0]['type'],rule_index=0)],
                unmet_requirements=[],scene=scene)


def reject(status):
    return dict(status=status,reason="请说明具体监测条件" if status=='needs_clarification' else "当前不能执行该需求",
                summary="",requirements=[],unmet_requirements=["安全帽识别"] if status=='unsupported' else [],scene=None)


@pytest.mark.parametrize('text',['','   ','你好！','hello','123456','!!!!','哈哈哈哈','a'*2001])
def test_obvious_invalid_text_does_not_call_api_or_fall_back(text):
    client=Client()
    with pytest.raises(RequirementRejected,match="需求"):
        resolve_scene_with_status(mode='deepseek',requirement=text,template_id='fire_detection',client=client)
    assert not client.calls


@pytest.mark.parametrize('status',['invalid_requirement','needs_clarification','unsupported'])
def test_semantic_rejection_does_not_retry_or_silently_use_template(status):
    client=Client(reject(status))
    with pytest.raises(RequirementRejected) as exc:
        resolve_scene_with_status(mode='deepseek',requirement='检测未戴安全帽',template_id='fire_detection',client=client)
    assert exc.value.status==status and len(client.calls)==1


@pytest.mark.parametrize('template,text',[
    ('on_duty','岗位无人5秒后报警'),('drowsiness','连续闭眼2秒后报警'),
    ('border_intrusion','有人进入区域报警'),('fire_detection','连续监测到火焰8秒后报警'),
])
def test_supported_requirements_preserve_current_scenes(template,text):
    payload=decision(text,template)
    if template=='fire_detection':
        payload['scene']['rules'][0]['params'].update(seconds=8)
    client=Client(payload)
    result=resolve_scene_with_status(mode='deepseek',requirement=text,client=client,video_fps=25,video_size=(640,480))
    assert result.scene.model_id==payload['scene']['model_id']
    assert result.requirement_summary==text
    assert 'Actual video dimensions are 640x480' in client.calls[0][0]['content']
    assert 'invalid_requirement' in client.calls[0][0]['content']
    if template=='fire_detection': assert result.scene.rules[0].params['frames']==201


@pytest.mark.parametrize('mutation',['bare','unmet','wrong_rule','invented_quote','missing_summary','out_of_bounds'])
def test_inconsistent_assessment_fails_closed(mutation):
    text='岗位无人5秒报警'
    payload=decision(text)
    if mutation=='bare': payload=payload['scene']
    elif mutation=='unmet': payload['unmet_requirements']=['安全帽检测']
    elif mutation=='wrong_rule': payload['requirements'][0]['capability']='eyes_closed_duration'
    elif mutation=='invented_quote': payload['requirements'][0]['source_text']='火灾检测'
    elif mutation=='missing_summary': payload['summary']=''
    elif mutation=='out_of_bounds': payload['scene']['rules'][0]['params']['polygon'][0]=[5000,5000]
    parser=SceneParser(Client(payload,payload),ModelRegistry.from_config(ROOT/'config/models.yaml'))
    with pytest.raises(SceneParseError): parser.parse_requirement(text,video_size=(640,480))


def test_invalid_json_can_be_repaired_once():
    text='岗位无人5秒报警'
    parser=SceneParser(Client('bad json',decision(text)),ModelRegistry.from_config(ROOT/'config/models.yaml'))
    scene,summary=parser.parse_requirement(text)
    assert scene.scene_type=='on_duty' and summary==text


def test_smoking_is_not_mistaken_for_fire_duration_before_semantic_check():
    client=Client(reject('unsupported'))
    with pytest.raises(RequirementRejected):
        resolve_scene_with_status(mode='deepseek',requirement='人员吸烟持续8秒后报警',client=client)
    assert len(client.calls)==1


def test_web_rejection_clears_previous_config_and_persists_until_edit(monkeypatch):
    import src.ui.app_service as service
    monkeypatch.setattr(service,'DeepSeekClient',lambda **kwargs: Client(reject('unsupported')))
    app=AppTest.from_file(ROOT/'app.py').run()
    app.button(key='prepare_scene').click().run()
    assert app.session_state['scene_spec'] is not None
    app.segmented_control(key='scene_mode').set_value('DeepSeek 解析').run()
    app.text_area(key='scene_requirement').set_value('检测未戴安全帽').run()
    app.button(key='prepare_scene').click().run()
    assert not app.exception
    assert app.session_state['scene_spec'] is None
    assert app.button(key='run_analysis').disabled
    assert any('现有模型或规则无法完整满足' in x.value for x in app.warning)
    app.run()
    assert app.session_state['scene_feedback']['status']=='unsupported'
    app.text_area(key='scene_requirement').set_value('岗位无人5秒报警').run()
    assert app.session_state['scene_feedback'] is None


def test_web_displays_validated_requirement_summary(monkeypatch):
    import src.ui.app_service as service
    text='连续闭眼2秒后报警'
    monkeypatch.setattr(service,'DeepSeekClient',lambda **kwargs: Client(decision(text,'drowsiness')))
    app=AppTest.from_file(ROOT/'app.py').run()
    app.segmented_control(key='scene_mode').set_value('DeepSeek 解析').run()
    app.text_area(key='scene_requirement').set_value(text).run()
    app.button(key='prepare_scene').click().run()
    assert not app.exception
    assert app.session_state['scene_spec']['scene_type']=='drowsiness'
    assert any('需求理解（请核对）' in x.value for x in app.info)

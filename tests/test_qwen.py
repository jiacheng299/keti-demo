import json
from pathlib import Path
import re
from urllib.error import HTTPError

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.llm.qwen_client import QwenClient, QwenSettings, QwenError, load_qwen_settings, save_qwen_settings
from src.pipeline.qwen_pipeline import parse_decision, run_qwen_video, video_plan
from src.security.qwen_budget import QwenBudget
from src.security.credential_store import save_api_key

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def video(tmp_path):
    path = tmp_path/'sample.avi'
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((48,64,3),i*10,dtype=np.uint8))
    writer.release()
    return path


class FakeVisual:
    settings = QwenSettings()

    def __init__(self, statuses=('satisfied','satisfied')):
        self.statuses = list(statuses)
        self.calls = 0

    def complete(self, content):
        self.calls += 1
        ids = [int(re.search(r'frame_id=(\d+)', x['text']).group(1)) for x in content if x['type']=='text' and x['text'].startswith('frame_id=')]
        return json.dumps({'status': self.statuses.pop(0),'evidence_frame_ids':[ids[0],ids[-1]],'reason':'画面可见动作','limitations':'仅采样画面'}), {'total_tokens':30}


def test_qwen_client_preserves_thinking_mode_and_returns_final_only():
    requests=[]
    def transport(request, timeout):
        requests.append(json.loads(request.data))
        return json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'OK','reasoning_content':'private reasoning'}}], 'usage':{'total_tokens':20}}).encode()
    assert QwenClient(api_key='test-only',transport=transport).complete('hello')[0]=='OK'
    assert 'enable_thinking' not in requests[0]
    assert requests[0]['model']=='qwen3-vl-32b-thinking'


@pytest.mark.parametrize('finish,content',[('length','partial'),('stop',''),('content_filter',None)])
def test_incomplete_outputs_never_count_as_negative(finish,content):
    def transport(*_):
        return json.dumps({'choices':[{'finish_reason':finish,'message':{'content':content}}]}).encode()
    with pytest.raises(QwenError):
        QwenClient(api_key='test-only',transport=transport).complete('hello')


def test_provider_error_is_redacted():
    def transport(request,timeout):
        raise HTTPError(request.full_url,401,'test-secret private details',{},None)
    with pytest.raises(QwenError) as exc:
        QwenClient(api_key='test-secret',transport=transport).complete('hello')
    assert '401' in str(exc.value) and 'test-secret' not in str(exc.value)


@pytest.mark.parametrize('url',['http://dashscope.aliyuncs.com/compatible-mode/v1','https://evil.example/compatible-mode/v1','https://dashscope.aliyuncs.com.evil.example/compatible-mode/v1','https://user:secret@dashscope.aliyuncs.com/compatible-mode/v1'])
def test_credentials_only_go_to_approved_official_addresses(url):
    with pytest.raises(ValueError): QwenSettings(base_url=url)


def test_settings_persist_without_credentials():
    settings=QwenSettings(model='qwen3-vl-plus',daily_requests=12,public_enabled=False)
    save_qwen_settings(settings)
    assert load_qwen_settings()==settings


@pytest.mark.parametrize('status,ids',[('satisfied',[]),('satisfied',[999]),('uncertain',[999]),('not_satisfied',[1,1])])
def test_evidence_must_reference_actual_input_frames(status,ids):
    with pytest.raises(QwenError):
        parse_decision(json.dumps(dict(status=status,evidence_frame_ids=ids,reason='test',limitations='test')),{1,2})


def test_budget_and_concurrency_are_shared_across_instances(tmp_path):
    path=tmp_path/'shared.sqlite'
    one,two=QwenBudget(2,path),QwenBudget(2,path)
    with one.session():
        with pytest.raises(QwenError,match='正在运行'):
            with two.session(): pass
        one.reserve()
        two.reserve()
        with pytest.raises(QwenError,match='额度'): one.reserve()
    with two.session(): assert two.used()==2


def test_full_inspection_review_evidence_exports_and_cache(video,tmp_path):
    client=FakeVisual()
    report=run_qwen_video(video,'检测明显转头动作',client=client,root=tmp_path/'results')
    assert report['status']=='completed' and report['requests']==2 and client.calls==2
    assert report['windows'][0]['reviewed']
    assert len(report['events'])==1
    assert all(0<=f['timestamp']<2 and Path(f['path']).is_file() for f in report['evidence'])
    capture=cv2.VideoCapture(report['result_video_path'])
    count=0
    while capture.read()[0]: count+=1
    capture.release()
    assert count==20
    assert (Path(report['run_dir'])/'windows.csv').is_file()
    cached=run_qwen_video(video,'检测明显转头动作',client=client,root=tmp_path/'results')
    assert cached['cached'] and client.calls==2


def test_disagreeing_review_is_uncertain(video,tmp_path):
    report=run_qwen_video(video,'检测明显转头动作',client=FakeVisual(['satisfied','not_satisfied']),root=tmp_path/'results')
    assert report['windows'][0]['status']=='uncertain'
    assert not report['events']


def test_failed_calls_do_not_cache_a_result(video,tmp_path):
    client=FakeVisual()
    def fail(*_): raise QwenError('timeout')
    client.complete=fail
    root=tmp_path/'results'
    with pytest.raises(QwenError): run_qwen_video(video,'检测明显转头动作',client=client,root=root)
    assert not list(root.glob('*.json'))
    assert len(list(root.glob('*/failure.json')))==1


def test_long_video_is_rejected_before_api(monkeypatch,video):
    from src.video.video_source import VideoSource,VideoMetadata
    monkeypatch.setattr(VideoSource,'metadata',lambda self: VideoMetadata(64,48,10,610,61))
    with pytest.raises(QwenError,match='60秒'): video_plan(video)


def test_web_fallback_requires_visual_eligibility_and_clears_on_video_change(monkeypatch,video):
    import src.ui.app_service as service
    response=dict(status='unsupported',reason='本地不支持转头动作',summary='转头检测',requirements=[],unmet_requirements=['转头'],scene=None,vlm_eligible=True,vlm_reason='可从图像检查转头')
    class Client:
        def complete(self,messages): return json.dumps(response)
    monkeypatch.setattr(service,'DeepSeekClient',lambda **kwargs:Client())
    app=AppTest.from_file(ROOT/'app.py').run()
    app.segmented_control(key='scene_mode').set_value('DeepSeek 解析').run()
    app.text_area(key='scene_requirement').set_value('检测明显转头动作').run()
    app.button(key='prepare_scene').click().run()
    assert app.button(key='run_qwen').disabled
    app.file_uploader[0].set_value((video.name,video.read_bytes(),'video/x-msvideo')).run()
    assert not app.button(key='run_qwen').disabled
    assert app.button(key='run_analysis').disabled
    app.session_state['qwen_result']={'obsolete':True}
    app.text_area(key='scene_requirement').set_value('发现异常就报警').run()
    assert app.session_state['qwen_result'] is None
    assert not any(b.key=='run_qwen' for b in app.button)
    assert not app.exception


def test_web_connection_test_uses_saved_qwen_key(monkeypatch):
    save_api_key('qwen-test-only','qwen')
    calls=[]
    def transport(request,timeout):
        calls.append(request.headers['Authorization'])
        return json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'OK'}}]}).encode()
    monkeypatch.setattr('src.llm.qwen_client._transport',transport)
    app=AppTest.from_file(ROOT/'app.py').run()
    app.button(key='test_qwen_connection').click().run()
    assert calls==['Bearer qwen-test-only'] and not app.exception
    assert any('连接成功' in s.value for s in app.success)


def test_public_cannot_change_settings_or_test_connection(monkeypatch):
    monkeypatch.setenv('DEMO_SHARED_INSTANCE','1')
    app=AppTest.from_file(ROOT/'app.py').run()
    assert not any(b.key=='test_qwen_connection' for b in app.button)
    assert not any(t.label=='Qwen Base URL' for t in app.text_input)


def test_non_visual_or_invalid_requirements_never_offer_cloud_route(monkeypatch):
    import src.ui.app_service as service
    class Client:
        def complete(self,messages):
            return json.dumps(dict(status='unsupported',reason='不能发送短信',summary='检测后发短信',requirements=[],
                                   unmet_requirements=['短信'],scene=None,vlm_eligible=False,vlm_reason='需要其他业务系统'))
    monkeypatch.setattr(service,'DeepSeekClient',lambda **kwargs:Client())
    app=AppTest.from_file(ROOT/'app.py').run()
    app.segmented_control(key='scene_mode').set_value('DeepSeek 解析').run()
    app.text_area(key='scene_requirement').set_value('发现有人时发短信').run()
    app.button(key='prepare_scene').click().run()
    assert not any(b.key=='run_qwen' for b in app.button)
    assert app.button(key='run_analysis').disabled


def test_public_owner_can_disable_cloud_route(monkeypatch,video):
    save_qwen_settings(QwenSettings(public_enabled=False))
    monkeypatch.setenv('DEMO_SHARED_INSTANCE','1')
    import src.ui.app_service as service
    class Client:
        def complete(self,messages):
            return json.dumps(dict(status='unsupported',reason='转头不支持',summary='检测转头',requirements=[],
                                   unmet_requirements=['转头'],scene=None,vlm_eligible=True,vlm_reason='可尝试视觉检验'))
    monkeypatch.setattr(service,'DeepSeekClient',lambda **kwargs:Client())
    app=AppTest.from_file(ROOT/'app.py').run()
    app.file_uploader[0].set_value((video.name,video.read_bytes(),'video/x-msvideo')).run()
    app.segmented_control(key='scene_mode').set_value('DeepSeek 解析').run()
    app.text_area(key='scene_requirement').set_value('检测明显转头动作').run()
    app.button(key='prepare_scene').click().run()
    assert not any(b.key=='run_qwen' for b in app.button)
    assert any('已关闭公网' in i.value for i in app.info)


def test_result_presentation_contains_report_evidence_and_bounded_previews(monkeypatch,video,tmp_path):
    import src.ui.app_service as service
    report=run_qwen_video(video,'检测明显转头动作',client=FakeVisual(),root=tmp_path/'results')
    class Client:
        def complete(self,messages):
            return json.dumps(dict(status='unsupported',reason='转头不支持',summary='检测转头',requirements=[],
                                   unmet_requirements=['转头'],scene=None,vlm_eligible=True,vlm_reason='可尝试视觉检验'))
    monkeypatch.setattr(service,'DeepSeekClient',lambda **kwargs:Client())
    monkeypatch.setattr('src.ui.qwen_view.run_qwen_video',lambda *args,**kwargs:report)
    app=AppTest.from_file(ROOT/'app.py').run()
    app.file_uploader[0].set_value((video.name,video.read_bytes(),'video/x-msvideo')).run()
    app.segmented_control(key='scene_mode').set_value('DeepSeek 解析').run()
    app.text_area(key='scene_requirement').set_value('检测明显转头动作').run()
    app.button(key='prepare_scene').click().run()
    app.button(key='run_qwen').click().run()
    assert not app.exception
    assert app.session_state['qwen_result']['events']
    assert any(s.value=='Qwen 检验结果' for s in app.subheader)
    assert any(b.label=='下载 Qwen 报告' for b in app.get('download_button'))
    app.file_uploader[0].set_value(('another.avi',video.read_bytes(),'video/x-msvideo')).run()
    assert app.session_state['qwen_result'] is None

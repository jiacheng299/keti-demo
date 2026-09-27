import pytest

from src.llm.scene_parser import SceneParser
from src.models.model_registry import ModelRegistry
from src.rules.base_rule import FrameState
from src.rules.drowsiness_rules import EyesClosedRule
from src.rules.occupancy_rules import UnderstaffedRule
from src.rules.rule_engine import RuleEngine
from src.schemas.detection import Detection
from src.schemas.face_observation import FaceObservation


POLYGON = [[0, 0], [100, 0], [100, 100], [0, 100]]
PERSON = Detection(frame_id=0, class_name="person", confidence=0.9, bbox_xyxy=(10,10,50,80))
CAR = PERSON.model_copy(update={"class_name": "car"})


def state(t, face=None, fps=10):
    return FrameState(frame_id=round(t*fps), timestamp_seconds=t, effective_fps=fps, face=face)


def face(ear=0.1, **kwargs):
    return FaceObservation(valid=True, bbox_xyxy=(10,10,210,300), left_ear=ear, right_ear=ear, **kwargs)


def test_post_starts_empty_emits_once_and_confirms_return():
    rule = UnderstaffedRule("post", POLYGON, seconds=1, startup_seconds=0.5, recovery_seconds=0.5)
    events=[]
    for i in range(31):
        events += rule.evaluate([CAR], state(i/10))
    assert [(e.event_type,e.timestamp_seconds) for e in events] == [("post_unstaffed",1.5)]
    assert events[0].confidence is None and events[0].track_id is None
    assert rule.progress[0].status == "confirmed"
    for i in range(31,40):
        events += rule.evaluate([PERSON], state(i/10))
    assert [(e.event_type,e.timestamp_seconds) for e in events][-1] == ("post_recovered",3.6)
    assert rule.progress[0].status == "normal"


def test_post_short_miss_does_not_accumulate_across_return_or_data_gap():
    rule=UnderstaffedRule("post",POLYGON,seconds=1,startup_seconds=0)
    for t, ds in [(0,[]),(.4,[]),(.5,[PERSON]),(.6,[]),(1,[]),(1.1,[PERSON]),(5,[]),(5.4,[])]:
        assert rule.evaluate(ds,state(t)) == []
    assert rule.progress[0].current == pytest.approx(.4)


def test_multiple_posts_count_people_independently_and_ignore_cooldown_for_recovery():
    registry=ModelRegistry.from_config("config/models.yaml")
    scene=SceneParser(None,registry).load_template("on_duty")
    scene.rules[0].params.update(polygon=POLYGON,min_count=2,seconds=.2,startup_seconds=0,recovery_seconds=.1)
    second=scene.rules[0].model_copy(deep=True)
    second.params.update(region_id="other",min_count=1)
    scene.rules.append(second)
    scene.alert.cooldown_seconds=3600
    engine=RuleEngine()
    events=[]
    for i in range(4): events+=engine.evaluate(scene,[PERSON],state(i/10))
    assert len(events)==1 and events[0].trigger_rule=="duty_zone"
    assert engine.progress[1].status=="normal"
    another=PERSON.model_copy(update={"bbox_xyxy":(55,10,80,80)})
    for i in range(4,7): events+=engine.evaluate(scene,[PERSON,another],state(i/10))
    for i in range(7,11): events+=engine.evaluate(scene,[PERSON],state(i/10))
    assert [e.event_type for e in events]==["post_unstaffed","post_recovered","post_unstaffed"]


@pytest.mark.parametrize("fps",[5,10,20])
def test_eye_closure_uses_seconds_once_per_episode_and_recovers(fps):
    rule=EyesClosedRule("eyes",seconds=2,recovery_seconds=.5)
    events=[]
    for i in range(fps*4):
        events+=rule.evaluate([],state(i/fps,face(),fps))
    assert len(events)==1 and events[0].timestamp_seconds==2
    for i in range(fps*4,fps*6):
        events+=rule.evaluate([],state(i/fps,face(.3),fps))
    assert [e.event_type for e in events]==["drowsiness_suspected","eyes_reopened"]
    assert rule.progress[0].status=="normal"


@pytest.mark.parametrize("invalid",[
    FaceObservation(), face(yaw_degrees=40),
    face().model_copy(update={"bbox_xyxy":(10,10,50,80)}),
    FaceObservation(reason="多人"),
])
def test_unknown_face_never_advances_closed_time(invalid):
    rule=EyesClosedRule("eyes",seconds=1)
    for i in range(6): assert not rule.evaluate([],state(i/10,face()))
    assert not rule.evaluate([],state(.6,invalid))
    assert rule.progress[0].status=="unknown"
    for i in range(7,15): assert not rule.evaluate([],state(i/10,face()))


def test_blink_single_eye_ambiguity_and_target_change_do_not_accumulate():
    rule=EyesClosedRule("eyes",seconds=1)
    single=face().model_copy(update={"right_ear":.3})
    for i in range(40):
        sample = single if i%6==0 else face(.22) if i%6==1 else face()
        assert not rule.evaluate([],state(i/10,sample))
    assert not rule.evaluate([],state(4,face(continuous=False)))
    assert rule.progress[0].current==0
    assert not rule.evaluate([],state(7,face()))
    assert rule.progress[0].current==0


@pytest.mark.parametrize("template,changes",[
    ("on_duty",{"min_count":0}), ("on_duty",{"startup_seconds":-1}),
    ("on_duty",{"seconds":float('nan')}), ("on_duty",{"recovery_seconds":0}),
    ("drowsiness",{"closed_ear":.3,"open_ear":.2}),
    ("drowsiness",{"min_face_width":0}), ("drowsiness",{"max_yaw_degrees":90}),
])
def test_new_templates_validate_parameter_ranges(template,changes):
    parser=SceneParser(None,ModelRegistry.from_config("config/models.yaml"))
    payload=parser.load_template(template).model_dump()
    payload['rules'][0]['params'].update(changes)
    with pytest.raises(ValueError): parser._validate_payload(payload)

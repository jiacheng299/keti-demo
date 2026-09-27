from src.rules.base_rule import FrameState
from src.rules.rule_engine import RuleEngine
from src.schemas.scene_spec import SceneSpec
from src.schemas.detection import Detection


def detection(x=5,kind="person"):
    return Detection(bbox_xyxy=(x,5,x+2,7),class_name=kind,confidence=.9,frame_id=0,track_id=7)


def state(t):
    return FrameState(frame_id=int(t*10),timestamp_seconds=t,effective_fps=10)


def scene(kind,params,**alert):
    return SceneSpec(scene_type="fire" if kind=="consecutive_frames" else "border",model_id="fire_smoke" if kind=="consecutive_frames" else "yolo_general",targets=["fire"] if kind=="consecutive_frames" else ["person"],rules=[{"type":kind,"params":params}],alert=alert)


def test_dwell_progress_uses_same_clock_and_resets_on_exit():
    engine=RuleEngine()
    spec=scene("dwell",{"polygon":[[0,0],[20,0],[20,20],[0,20]],"seconds":5})
    engine.evaluate(spec,[detection()],state(0))
    assert not engine.evaluate(spec,[detection()],state(3.6))
    assert engine.progress[0].current == 3.6
    assert engine.progress[0].required == 5
    assert engine.evaluate(spec,[detection()],state(5))
    assert engine.progress[0].status == "confirmed"
    engine.evaluate(spec,[detection(30)],state(6))
    assert engine.progress[0].current == 0
    assert "区域外" in engine.progress[0].reason


def test_fire_progress_tracks_hits_gaps_reset_and_confirmation():
    engine=RuleEngine()
    spec=scene("consecutive_frames",{"frames":3,"max_gap_frames":1,"seconds":.2})
    engine.evaluate(spec,[detection(kind="fire")],state(0))
    engine.evaluate(spec,[detection(kind="fire")],state(.1))
    assert engine.progress[0].current == .1
    engine.evaluate(spec,[],state(.2))
    assert "漏检 1/1" in engine.progress[0].reason
    assert engine.progress[0].current == .1
    events=engine.evaluate(spec,[detection(kind="fire")],state(.3))
    assert any(e.event_type=="fire_confirmed" for e in events)
    assert engine.progress[0].current == .2
    engine.evaluate(spec,[],state(.4))
    engine.evaluate(spec,[],state(.5))
    assert engine.progress[0].current == 0
    assert "清零" in engine.progress[0].reason


def test_region_progress_explains_first_sighting_cooldown_and_disabled_alerts():
    engine=RuleEngine()
    spec=scene("enter_region",{"polygon":[[0,0],[20,0],[20,20],[0,20]]},cooldown_seconds=10)
    engine.evaluate(spec,[detection()],state(0))
    assert "首次观察" in engine.progress[0].reason
    engine.evaluate(spec,[detection(30)],state(1))
    assert engine.evaluate(spec,[detection()],state(2))
    engine.evaluate(spec,[detection(30)],state(3))
    assert not engine.evaluate(spec,[detection()],state(4))
    assert engine.progress[0].status == "cooldown"
    assert "8.0 秒" in engine.progress[0].reason
    spec.alert.enabled=False
    assert not engine.evaluate(spec,[detection()],state(5))
    assert "报警已关闭" in engine.progress[0].reason

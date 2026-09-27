"""Editable configuration cards and a local, dependency-free polygon canvas."""

import base64
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from src.llm.scene_parser import SceneParser
from src.models.model_registry import ModelRegistry
from src.schemas.scene_time import resolve_time_rules
from src.ui.app_service import MODEL_CONFIG_PATH
from src.visualization.scene_overlay import TARGET_LABELS


canvas_component = components.declare_component("region_canvas", path=str(Path(__file__).parent / "region_canvas"))
RULE_LABELS = {"enter_region": "进入禁区", "leave_region": "离开禁区", "dwell": "区域滞留", "consecutive_frames": "持续检测确认", "region_understaffed": "岗位持续缺员", "eyes_closed_duration": "持续闭眼"}


@st.cache_data(show_spinner=False)
def preview_frame(path: str, frame_index: int) -> tuple[str, int, int]:
    capture = cv2.VideoCapture(path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError("无法读取预览帧")
        h, w = frame.shape[:2]
        ok, data = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("无法生成预览图")
        return "data:image/jpeg;base64," + base64.b64encode(data).decode(), w, h
    finally:
        capture.release()


def validate_geometry(points, width: int, height: int) -> None:
    SceneParser._validate_polygon(points)
    if any(x >= width or y >= height for x, y in points):
        raise ValueError("区域超出视频边界，请重新绘制或调整顶点")
    if len(set(tuple(p) for p in points)) != len(points):
        raise ValueError("区域存在重复顶点")
    contour = np.asarray(points, dtype=np.float32)
    if abs(cv2.contourArea(contour)) < 4:
        raise ValueError("区域面积太小，请重新绘制")
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def intersect(a,b,c,d):
        # Include touching and collinear overlaps for nonadjacent edges.
        return (max(min(a[0],b[0]),min(c[0],d[0])) <= min(max(a[0],b[0]),max(c[0],d[0]))
                and max(min(a[1],b[1]),min(c[1],d[1])) <= min(max(a[1],b[1]),max(c[1],d[1]))
                and cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0)
    n = len(points)
    for i in range(n):
        for j in range(i+1,n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            if intersect(points[i],points[(i+1)%n],points[j],points[(j+1)%n]):
                raise ValueError("多边形边线不能相交，请调整顶点")


def render_scene_editor(video_path, metadata) -> bool:
    """Return whether draft changes/invalid regions must block analysis."""
    scene = st.session_state["scene_spec"]
    if "cfg_draft" not in st.session_state:
        st.session_state["cfg_draft"] = deepcopy(scene)
        st.session_state["cfg_revision"] = uuid4().hex
    draft = st.session_state["cfg_draft"]
    revision = st.session_state["cfg_revision"]
    registry = ModelRegistry.from_config(MODEL_CONFIG_PATH)
    definition = registry.get(draft["model_id"])
    if st.session_state.pop("cfg_applied_notice", False):
        st.success("配置已应用，可以开始分析。")
    st.markdown("**配置卡片**")
    st.caption(f"模型：{draft['model_id']} · 检测置信度阈值：{definition.confidence:g} · 修改后点击“应用配置”，无需再次调用 DeepSeek。")
    monitoring = draft["scene_type"] in {"on_duty", "drowsiness"}
    draft["targets"] = st.multiselect("监测目标", ["person"] if monitoring else definition.classes, default=draft["targets"], format_func=lambda v: TARGET_LABELS.get(v,v), key="cfg_targets", disabled=monitoring)
    if draft["scene_type"] == "border":
        st.caption("所有区域共用这些监测目标，可选择人员、车辆或两者。所有支持的车辆类型统一按 car 监测和记录。")
    cols = st.columns(2)
    draft["alert"]["enabled"] = cols[0].checkbox("启用报警", value=draft["alert"]["enabled"], key="cfg_enabled")
    if monitoring:
        cols[1].caption("每次异常只记录一次，恢复后可再次报警。")
    else:
        draft["alert"]["cooldown_seconds"] = cols[1].number_input("同类事件冷却时间（秒）", min_value=0.0, max_value=3600.0, value=float(draft["alert"]["cooldown_seconds"]), key="cfg_cooldown")
    index = st.selectbox("编辑规则", range(len(draft["rules"])), format_func=lambda i: f"规则 {i+1} · {RULE_LABELS[draft['rules'][i]['type']]}", key=f"cfg_selected_{revision}")
    rule = draft["rules"][index]
    params = rule["params"]
    prefix = f"cfg_rule_{revision}_{index}"
    width, height = (metadata.width, metadata.height) if metadata else (640,480)
    pending = False
    with st.container(border=True):
        if draft["scene_type"] in {"border", "on_duty"}:
            duty = draft["scene_type"] == "on_duty"
            if duty:
                params["min_count"] = st.number_input("岗位最低人数", min_value=1, max_value=100, value=int(params.get("min_count",1)), key=prefix+"_count")
                params["seconds"] = st.number_input("连续缺员报警（秒）", min_value=0.1, max_value=3600.0, value=float(params.get("seconds",5)), key=prefix+"_seconds")
                params["recovery_seconds"] = st.number_input("回岗确认（秒）", min_value=0.1, max_value=3600.0, value=float(params.get("recovery_seconds",1)), key=prefix+"_recovery")
                params["startup_seconds"] = st.number_input("启动宽限（秒）", min_value=0.0, max_value=3600.0, value=float(params.get("startup_seconds",2)), key=prefix+"_startup")
                st.caption("按检测框中心统计岗位人数；从视频开始就无人也会计时。请把区域画在工作位上，避开过道。此功能不识别员工身份。")
            else:
                rule["type"] = st.selectbox("触发条件", ["enter_region","leave_region","dwell"], index=["enter_region","leave_region","dwell"].index(rule["type"]), format_func=RULE_LABELS.get, key=prefix+"_type")
                if rule["type"] == "dwell":
                    params["seconds"] = st.number_input("滞留达到（秒）", min_value=0.1, max_value=3600.0, value=float(params.get("seconds",5)), key=prefix+"_seconds")
                else:
                    params.pop("seconds",None)
                st.caption("按检测框中心判断内外；进入/离开需要同一目标 ID 的内外变化，首次出现在区域内不算进入。滞留以视频时间和目标 ID 计时。")
            params["name"] = st.text_input("区域名称", value=params.get("name", f"{'岗位' if duty else '禁区'} {index+1}"), max_chars=40, key=prefix+"_name")
            if video_path and metadata:
                position = st.slider("选择绘图参考帧", 0, max(1,metadata.frame_count-1), 0, key=prefix+"_frame")
                data,w,h = preview_frame(str(video_path),position)
                token = f"{revision}:{index}:{video_path}"
                applied = scene["rules"][index]["params"].get("polygon",params["polygon"]) if index < len(scene["rules"]) else params["polygon"]
                value = canvas_component(image=data,width=w,height=h,points=params["polygon"],applied=applied,others=[r["params"]["polygon"] for i,r in enumerate(draft["rules"]) if i != index],token=token,key=prefix+"_canvas")
                if value and value.get("token") == token:
                    pending = bool(value.get("pending"))
                    if not pending:
                        params["polygon"] = value["points"]
                st.caption("黄色为当前区域，灰色为其他区域。矩形拖动绘制；多边形依次点选后完成；白色顶点可拖动。")
            else:
                st.info("上传视频后可在画面上绘制矩形、多边形，并拖动顶点。")
        elif draft["scene_type"] == "drowsiness":
            st.info("支持最多 5 张人脸，每人独立计时和报警。请确保每个人的眼睛清晰；遮挡、严重侧脸或人脸太小时，该人显示无法判断。")
            params["seconds"] = st.number_input("连续闭眼报警（秒）", min_value=0.1, max_value=3600.0, value=float(params.get("seconds",2)), key=prefix+"_seconds")
            params["recovery_seconds"] = st.number_input("睁眼恢复（秒）", min_value=0.1, max_value=3600.0, value=float(params.get("recovery_seconds",0.5)), key=prefix+"_recovery")
            with st.expander("眼部阈值与画面质量", expanded=False):
                st.caption("EAR 是眼睛高度与宽度的比例，越小表示越接近闭眼。阈值需根据拍摄条件调整。")
                params["closed_ear"] = st.number_input("闭眼 EAR 阈值", min_value=0.01, max_value=0.8, value=float(params.get("closed_ear",0.20)), step=0.01, key=prefix+"_closed")
                params["open_ear"] = st.number_input("睁眼 EAR 阈值", min_value=0.01, max_value=0.8, value=float(params.get("open_ear",0.24)), step=0.01, key=prefix+"_open")
                params["min_face_width"] = st.number_input("最小人脸宽度（像素）", min_value=40, max_value=2000, value=int(params.get("min_face_width",100)), key=prefix+"_width")
                params["max_yaw_degrees"] = st.number_input("允许的最大侧脸角度", min_value=5.0, max_value=45.0, value=float(params.get("max_yaw_degrees",30)), key=prefix+"_yaw")
        else:
            unit = st.radio("确认阈值单位", ["秒","命中帧"], index=0 if "seconds" in params else 1, horizontal=True, key=prefix+"_unit")
            if unit == "秒":
                params["seconds"] = st.number_input("持续检测达到（秒）", min_value=0.1,max_value=3600.0,value=float(params.get("seconds",8)),key=prefix+"_seconds")
            else:
                params.pop("seconds",None)
                params["frames"] = st.number_input("确认所需命中帧数", min_value=1,max_value=1000000,value=int(params.get("frames",8)),key=prefix+"_frames")
            params["max_gap_frames"] = st.number_input("允许连续漏检帧数",min_value=0,max_value=100,value=int(params.get("max_gap_frames",0)),key=prefix+"_gap")
            st.caption("严格连续检测请设为 0。容忍漏检时，漏检帧不增加命中进度，实际确认会延后。")
    if draft["scene_type"] in {"border", "on_duty"}:
        duty = draft["scene_type"] == "on_duty"
        st.caption("每个岗位独立统计人数、缺员与回岗时间。添加后请切换到新规则并绘制岗位区域。" if duty else "添加区域规则：增加一块独立禁区及其触发条件。例如，区域 A 有目标进入就报警，区域 B 停留超过 5 秒才报警。添加后在“编辑规则”中选择新规则，再绘制区域并设置条件。")
        cols=st.columns(2)
        if cols[0].button("添加区域规则",key="cfg_add",disabled=len(draft["rules"])>=20):
            added = {"type":"region_understaffed" if duty else "enter_region","params":{"region_id":"zone_"+uuid4().hex[:10],"name":f"{'岗位' if duty else '禁区'} {len(draft['rules'])+1}","polygon":[[width*.25,height*.25],[width*.75,height*.25],[width*.75,height*.75],[width*.25,height*.75]]}}
            if duty:
                added["params"].update(min_count=1, seconds=5, recovery_seconds=1, startup_seconds=2)
            draft["rules"].append(added)
            st.session_state["cfg_revision"]=uuid4().hex
            st.rerun()
        if cols[1].button("删除当前区域规则",key="cfg_delete",disabled=len(draft["rules"])<=1):
            draft["rules"].pop(index)
            st.session_state["cfg_revision"]=uuid4().hex
            st.rerun()
    error = None
    try:
        validated = SceneParser(None,registry)._validate_payload(draft)
        if metadata:
            for r in validated.rules:
                if "polygon" in r.params:
                    validate_geometry(r.params["polygon"],width,height)
            validated = resolve_time_rules(validated,metadata.fps)
        elif any(r.type == "consecutive_frames" and "seconds" in r.params for r in validated.rules):
            raise ValueError("请先上传视频，以计算持续时间阈值")
        payload=validated.model_dump(mode="json")
    except (ValueError,TypeError,KeyError) as exc:
        error=str(exc)
        payload=None
    if pending:
        error="正在绘制区域，请完成绘制或恢复已应用区域。"
    if error:
        st.warning(error)
    dirty = payload != scene or bool(error)
    if st.button("应用配置",type="primary",key="cfg_apply",disabled=bool(error)):
        st.session_state["scene_spec"]=payload
        st.session_state["cfg_draft"]=deepcopy(payload)
        st.session_state["run_result"]=None
        st.session_state["scene_uses_video_fps"]=False
        st.session_state["scene_video_fps"]=metadata.fps if metadata else None
        st.session_state["cfg_applied_notice"]=True
        st.rerun()
    elif dirty and not error:
        st.info("配置有未应用的修改，请先点击“应用配置”。")
    if payload and metadata and payload["scene_type"] == "fire":
        for r in payload["rules"]:
            st.caption(f"确认阈值：{r['params'].get('frames',3)} 个命中帧 · 无漏检时约 {(r['params'].get('frames',3)-1)/metadata.fps:g} 秒（{metadata.fps:g} FPS）")
    with st.expander("查看已应用配置 JSON",expanded=False):
        st.json(st.session_state["scene_spec"])
    return dirty

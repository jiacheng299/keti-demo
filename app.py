"""Single-page Streamlit interface for the local video-analysis demo."""

from pathlib import Path

from PIL import Image
import streamlit as st

from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.schemas.scene_spec import SceneSpec
from src.llm.requirement_guard import RequirementRejected
from src.ui.app_service import (
    MODEL_CONFIG_PATH,
    PROJECT_ROOT,
    TEMPLATE_OPTIONS,
    create_run_dir,
    event_rows_for_display,
    requires_video_fps,
    resolve_scene_with_status,
    save_uploaded_video,
    snapshot_paths_for_display,
)
from src.ui.api_settings import render_api_settings
from src.ui.qwen_view import render_qwen_route, render_qwen_result
from src.ui.scene_editor import render_scene_editor
from src.ui.progress_view import render_progress_view
from src.ui.share_access import is_shared_instance
from src.video.browser_video import make_browser_playable
from src.video.video_source import VideoSource


st.set_page_config(page_title="开放场景视觉语义认知", layout="wide")
st.session_state.setdefault("scene_spec", None)
st.session_state.setdefault("run_result", None)


def clear_scene_configuration() -> None:
    """Require a fresh configuration after changing the mode or its inputs."""
    st.session_state["scene_spec"] = None
    st.session_state["run_result"] = None
    st.session_state["scene_warning"] = None
    st.session_state["scene_source"] = None
    st.session_state["scene_uses_video_fps"] = False
    st.session_state["scene_video_fps"] = None
    st.session_state["scene_feedback"] = None
    st.session_state["requirement_summary"] = None
    st.session_state["qwen_result"] = None
    st.session_state.pop("qwen_evidence_page", None)
    for key in list(st.session_state):
        if key.startswith("cfg_"):
            del st.session_state[key]

st.title("开放场景视觉语义认知")


def clear_video_results():
    st.session_state["qwen_result"] = None
    st.session_state["run_result"] = None
    st.session_state.pop("qwen_evidence_page", None)

with st.container(border=True):
    st.subheader("1. 运行状态")
    api_key = render_api_settings()
    st.caption("模型仅在分析期间加载，分析结束后自动释放。")

with st.container(border=True):
    st.subheader("2. 上传视频")
    uploaded_file = st.file_uploader(
        "上传视频文件",
        type=["mp4", "avi", "mov", "ogv", "mkv"],
        max_upload_size=95 if is_shared_instance() else 500,
        key="video_upload",
        on_change=clear_video_results,
        help=f"支持 MP4、AVI、MOV、OGV 和 MKV，单文件不超过 {95 if is_shared_instance() else 500} MB。",
    )

    uploaded_video_path: Path | None = None
    uploaded_video_fps: float | None = None
    metadata = None
    if uploaded_file is not None:
        try:
            uploaded_video_path = save_uploaded_video(
                uploaded_file.getvalue(),
                uploaded_file.name,
                PROJECT_ROOT / "runs" / "uploads",
            )
            with VideoSource.open(uploaded_video_path) as source:
                metadata = source.metadata()
            uploaded_video_fps = metadata.fps
            metric_columns = st.columns(4)
            metric_columns[0].metric("分辨率", f"{metadata.width} × {metadata.height}")
            metric_columns[1].metric("帧率", f"{metadata.fps:.2f} FPS")
            metric_columns[2].metric("总帧数", str(metadata.frame_count))
            metric_columns[3].metric("时长", f"{metadata.duration_seconds:.1f} 秒")
        except Exception as error:
            st.error(f"视频不可用：{error}")
            uploaded_video_path = None

editor_dirty = False
with st.container(border=True):
    st.subheader("3. 场景配置")
    mode_label = st.segmented_control(
        "配置方式",
        ["离线模板", "DeepSeek 解析"],
        default="离线模板",
        key="scene_mode",
        on_change=clear_scene_configuration,
    )
    template_id = None
    requirement = ""
    if mode_label == "DeepSeek 解析":
        requirement = st.text_area(
            "场景需求",
            placeholder="例如：连续监测到火焰 8 秒后报警。",
            max_chars=2000,
            help="请描述监测对象与具体行为或状态，例如：岗位连续无人 5 秒报警。系统会先检查需求含义和现有模型能力。",
            key="scene_requirement",
            on_change=clear_scene_configuration,
        )
    else:
        template_label = st.selectbox(
            "离线模板",
            list(TEMPLATE_OPTIONS),
            key="template_label",
            on_change=clear_scene_configuration,
        )
        template_id = TEMPLATE_OPTIONS[template_label]

    if st.button(
        "生成场景配置",
        key="prepare_scene",
        icon=":material/settings_suggest:",
    ):
        clear_scene_configuration()
        if mode_label == "DeepSeek 解析" and not requirement.strip():
            st.error("使用 DeepSeek 解析时，请先填写场景需求。")
        else:
            try:
                with st.spinner("正在生成场景配置……"):
                    resolution = resolve_scene_with_status(
                        mode="deepseek" if mode_label == "DeepSeek 解析" else "offline",
                        requirement=requirement,
                        template_id=template_id,
                        api_key=api_key,
                        video_fps=uploaded_video_fps,
                        video_size=(metadata.width, metadata.height) if metadata else None,
                    )
                st.session_state["scene_spec"] = resolution.scene.model_dump(mode="json")
                if resolution.source == "离线模板" and resolution.scene.scene_type == "on_duty" and metadata:
                    # Template geometry is authored against the 360x240 office sample.
                    for rule in st.session_state["scene_spec"]["rules"]:
                        rule["params"]["polygon"] = [[min(metadata.width-1,x*metadata.width/360), min(metadata.height-1,y*metadata.height/240)] for x,y in rule["params"]["polygon"]]
                st.session_state["scene_source"] = resolution.source
                st.session_state["scene_warning"] = resolution.warning
                st.session_state["requirement_summary"] = resolution.requirement_summary
                st.session_state["scene_video_fps"] = uploaded_video_fps
                st.session_state["scene_uses_video_fps"] = (
                    resolution.source == "DeepSeek" and requires_video_fps(requirement)
                )
                st.session_state["run_result"] = None
            except RequirementRejected as error:
                st.session_state["scene_feedback"] = {
                    "status": error.status, "reason": str(error), "unmet": error.unmet,
                    "vlm_eligible": error.vlm_eligible, "vlm_reason": error.vlm_reason,
                }
            except Exception as error:
                st.session_state["scene_spec"] = None
                st.session_state["run_result"] = None
                st.error(f"场景配置生成失败：{error}")

    feedback = st.session_state.get("scene_feedback")
    if feedback:
        if feedback["status"] == "unsupported" and feedback.get("vlm_eligible"):
            render_qwen_route(uploaded_video_path, requirement, feedback)
        else:
            labels = {"invalid_requirement": "请输入场景监测需求", "needs_clarification": "需要补充需求", "unsupported": "现有模型或规则无法完整满足"}
            st.warning(f"{labels.get(feedback['status'], '需求未通过校验')}：{feedback['reason']}")
            for missing in feedback["unmet"]:
                st.write("未满足的要求：", missing)
            if feedback["status"] == "unsupported":
                render_qwen_route(uploaded_video_path, requirement, feedback)

    if st.session_state["scene_spec"] is not None:
        if st.session_state.get("scene_warning"):
            st.warning(st.session_state["scene_warning"])
        st.caption(f"配置来源：{st.session_state.get('scene_source', '已有配置')}")
        if st.session_state.get("requirement_summary"):
            st.info("需求理解（请核对）：" + st.session_state["requirement_summary"])
        editor_dirty = render_scene_editor(uploaded_video_path, metadata)

fps_changed = bool(st.session_state.get("scene_uses_video_fps")) and (
    st.session_state.get("scene_video_fps") != uploaded_video_fps
)
if fps_changed:
    st.warning("当前视频帧率与生成配置时不同，请重新生成场景配置，以保持持续时间准确。")
can_run = uploaded_video_path is not None and st.session_state["scene_spec"] is not None and not fps_changed and not editor_dirty
with st.container(border=True):
    st.subheader("4. 开始分析")
    run_analysis = st.button(
        "开始分析",
        key="run_analysis",
        type="primary",
        icon=":material/play_arrow:",
        disabled=not can_run,
    )
    if not can_run:
        if feedback and feedback.get("vlm_eligible"):
            st.caption("请使用上方“使用 Qwen 检验”入口。")
        else:
            st.caption("上传有效视频并生成场景配置后即可开始。")

    if run_analysis and uploaded_video_path is not None:
        progress_bar = st.progress(0.0, text="正在准备模型……")
        try:
            scene = SceneSpec.model_validate(st.session_state["scene_spec"])
            pipeline = AnalysisPipeline.from_model_config(MODEL_CONFIG_PATH)
            run_dir = create_run_dir()

            def update_progress(progress) -> None:
                progress_bar.progress(
                    progress.fraction,
                    text=(
                        f"已读取 {progress.frames_read}/{progress.total_frames} 帧，"
                        f"已推理 {progress.processed_frames} 帧，"
                        f"已记录 {progress.event_count} 个事件"
                    ),
                )

            summary = pipeline.run(
                uploaded_video_path,
                scene,
                run_dir,
                progress_callback=update_progress,
            )
            progress_bar.progress(1.0, text="正在生成浏览器兼容视频……")
            make_browser_playable(summary.result_video_path, summary.result_video_path)
            progress_bar.progress(1.0, text="分析完成")
            st.session_state["run_result"] = {
                "run_dir": str(summary.run_dir),
                "result_video_path": str(summary.result_video_path),
                "frames_read": summary.frames_read,
                "processed_frames": summary.processed_frames,
                "detection_count": summary.detection_count,
                "event_count": len(summary.events),
                "elapsed_seconds": summary.elapsed_seconds,
                "events": [event.model_dump(mode="json") for event in summary.events],
                "model_id": scene.model_id,
            }
            st.success("分析完成，模型已释放。")
        except Exception as error:
            st.error(f"分析失败：{error}")

render_qwen_result()
run_result = st.session_state["run_result"]
if run_result is not None:
    with st.container(border=True):
        st.subheader("5. 分析结果")
        result_columns = st.columns(4)
        result_columns[0].metric("读取帧数", str(run_result["frames_read"]))
        result_columns[1].metric("推理帧数", str(run_result["processed_frames"]))
        result_columns[2].metric("检测数量", str(run_result["detection_count"]))
        result_columns[3].metric("事件数量", str(run_result["event_count"]))
        st.caption(
            f"模型：{run_result['model_id']} · "
            f"耗时：{run_result['elapsed_seconds']:.2f} 秒"
        )

        result_video_path = Path(run_result["result_video_path"])
        run_dir = Path(run_result["run_dir"])
        if result_video_path.is_file():
            with st.container(horizontal=True, horizontal_alignment="center"):
                st.video(result_video_path, width=640)
        else:
            st.warning("结果视频文件不存在，请重新运行分析。")

        st.subheader("事件记录")
        if run_result["events"]:
            st.dataframe(event_rows_for_display(run_result["events"]), hide_index=True)
        else:
            st.info("本次分析未触发事件。")

        render_progress_view(run_dir)

        snapshots = snapshot_paths_for_display(run_dir, run_result["events"])
        if snapshots:
            st.subheader("事件截图")
            snapshot_columns = st.columns(min(3, len(snapshots)))
            for index, (snapshot_path, event) in enumerate(snapshots):
                with Image.open(snapshot_path) as snapshot:
                    image_width, image_height = snapshot.size
                # Fit both landscape and portrait previews without enlarging originals.
                preview_width = max(1, int(min(image_width, 320, 360 * image_width / image_height)))
                snapshot_columns[index % len(snapshot_columns)].image(
                    snapshot_path,
                    caption=(
                        f"{event['event_id']} · "
                        f"{event_rows_for_display([event])[0]['事件类型']}"
                    ),
                    width=preview_width,
                )

        with st.container(horizontal=True):
            if result_video_path.is_file():
                st.download_button(
                    "下载结果视频",
                    result_video_path.read_bytes(),
                    file_name="result.mp4",
                    mime="video/mp4",
                    key="download_video",
                    icon=":material/download:",
                )
            for filename, label, mime in [
                ("events.json", "下载事件 JSON", "application/json"),
                ("events.csv", "下载事件 CSV", "text/csv"),
                ("summary.json", "下载运行摘要", "application/json"),
                ("config.json", "下载场景配置", "application/json"),
                ("rule_progress.jsonl", "下载规则进度与原因", "application/x-ndjson"),
            ]:
                path = run_dir / filename
                if path.is_file():
                    st.download_button(
                        label,
                        path.read_bytes(),
                        file_name=filename,
                        mime=mime,
                        key=f"download_{filename}",
                        icon=":material/download:",
                    )

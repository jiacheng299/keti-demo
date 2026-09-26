"""Single-page Streamlit interface for the local video-analysis demo."""

import os
from pathlib import Path

import streamlit as st

from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.schemas.scene_spec import SceneSpec
from src.ui.app_service import (
    MODEL_CONFIG_PATH,
    PROJECT_ROOT,
    TEMPLATE_OPTIONS,
    create_run_dir,
    event_rows_for_display,
    resolve_scene_spec,
    save_uploaded_video,
    snapshot_paths_for_display,
)
from src.video.browser_video import make_browser_playable
from src.video.video_source import VideoSource


st.set_page_config(page_title="开放场景视觉语义认知 Demo", layout="wide")
st.session_state.setdefault("scene_spec", None)
st.session_state.setdefault("run_result", None)

st.title("开放场景视觉语义认知 Demo")
st.caption("上传本地视频，通过离线模板或 DeepSeek 生成场景配置，再使用 CPU 小模型完成分析。")

with st.container(border=True):
    st.subheader("1. 运行状态")
    if os.getenv("DEEPSEEK_API_KEY"):
        st.success("DeepSeek API：已配置")
    else:
        st.warning("DeepSeek API：未配置，将使用离线模板")
    st.caption("模型仅在分析期间加载，分析结束后自动释放。")

with st.container(border=True):
    st.subheader("2. 上传视频")
    uploaded_file = st.file_uploader(
        "上传视频文件",
        type=["mp4", "avi", "mov", "ogv", "mkv"],
        max_upload_size=500,
        key="video_upload",
        help="支持 MP4、AVI、MOV、OGV 和 MKV，单文件不超过 500 MB。",
    )

    uploaded_video_path: Path | None = None
    if uploaded_file is not None:
        try:
            uploaded_video_path = save_uploaded_video(
                uploaded_file.getvalue(),
                uploaded_file.name,
                PROJECT_ROOT / "runs" / "uploads",
            )
            with VideoSource.open(uploaded_video_path) as source:
                metadata = source.metadata()
            metric_columns = st.columns(4)
            metric_columns[0].metric("分辨率", f"{metadata.width} × {metadata.height}")
            metric_columns[1].metric("帧率", f"{metadata.fps:.2f} FPS")
            metric_columns[2].metric("总帧数", str(metadata.frame_count))
            metric_columns[3].metric("时长", f"{metadata.duration_seconds:.1f} 秒")
        except Exception as error:
            st.error(f"视频不可用：{error}")
            uploaded_video_path = None

with st.container(border=True):
    st.subheader("3. 场景配置")
    mode_label = st.segmented_control(
        "配置方式",
        ["离线模板", "DeepSeek 解析"],
        default="离线模板",
        key="scene_mode",
    )
    template_label = st.selectbox(
        "离线模板",
        list(TEMPLATE_OPTIONS),
        key="template_label",
        help="DeepSeek 不可用或输出无效时，也会回退到这里选择的模板。",
    )
    requirement = st.text_area(
        "场景需求",
        placeholder="例如：连续检测到火焰 8 帧后报警，允许 1 帧短暂漏检。",
        key="scene_requirement",
        disabled=mode_label != "DeepSeek 解析",
    )

    if st.button(
        "生成场景配置",
        key="prepare_scene",
        icon=":material/settings_suggest:",
    ):
        if mode_label == "DeepSeek 解析" and not requirement.strip():
            st.error("使用 DeepSeek 解析时，请先填写场景需求。")
        else:
            try:
                scene = resolve_scene_spec(
                    mode="deepseek" if mode_label == "DeepSeek 解析" else "offline",
                    requirement=requirement,
                    template_id=TEMPLATE_OPTIONS[template_label],
                )
                st.session_state["scene_spec"] = scene.model_dump(mode="json")
                st.session_state["run_result"] = None
            except Exception as error:
                st.error(f"场景配置生成失败：{error}")

    if st.session_state["scene_spec"] is not None:
        scene_payload = st.session_state["scene_spec"]
        st.caption(f"当前模型：{scene_payload['model_id']}")
        st.json(scene_payload, expanded=1)

can_run = uploaded_video_path is not None and st.session_state["scene_spec"] is not None
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
            st.video(result_video_path)
        else:
            st.warning("结果视频文件不存在，请重新运行分析。")

        st.subheader("事件记录")
        if run_result["events"]:
            st.dataframe(event_rows_for_display(run_result["events"]), hide_index=True)
        else:
            st.info("本次分析未触发事件。")

        snapshots = snapshot_paths_for_display(run_dir, run_result["events"])
        if snapshots:
            st.subheader("事件截图")
            snapshot_columns = st.columns(min(3, len(snapshots)))
            for index, (snapshot_path, event) in enumerate(snapshots):
                snapshot_columns[index % len(snapshot_columns)].image(
                    snapshot_path,
                    caption=(
                        f"{event['event_id']} · "
                        f"{event_rows_for_display([event])[0]['事件类型']}"
                    ),
                    width="stretch",
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

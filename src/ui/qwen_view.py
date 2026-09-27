"""Manual cloud route and evidence presentation for unsupported visual requirements."""
import io
from pathlib import Path
import zipfile

from PIL import Image
import streamlit as st

from src.llm.qwen_client import QwenError, load_qwen_settings
from src.pipeline.qwen_pipeline import run_qwen_video, video_plan, STATUS_NAMES
from src.security.credential_store import CredentialStoreError
from src.ui.share_access import is_shared_instance
from src.video.video_source import VideoSource


def render_qwen_route(path, requirement, feedback):
    if not feedback.get('vlm_eligible'):
        st.info(feedback.get('vlm_reason') or "该需求尚未确认适合云端视觉检验，请明确可从画面观察的条件后重新解析。")
        return
    st.info(
        "现有模板与您的需求匹配度不足，本地模型和规则无法完整满足本次监测要求。"
        "建议采用阿里云 Qwen 多模态大模型，结合您的需求分析视频采样画面，进行疑似事件监测。"
    )
    if feedback.get('reason'):
        st.text("未匹配原因：" + feedback['reason'])
    if feedback.get('unmet'):
        st.text("现有模板未覆盖的需求：" + "；".join(feedback['unmet']))
    if feedback.get('vlm_reason'):
        st.text("采用多模态分析的依据：" + feedback['vlm_reason'])
    try:
        settings = load_qwen_settings()
        if is_shared_instance() and not settings.public_enabled:
            st.info("分享者已关闭公网 Qwen 检验。")
            return
        st.caption(
            "点击“使用 Qwen 检验”后才会开始云端监测，将需求和视频采样画面发送至阿里云，"
            "云端调用会消耗所配置账号的 API 额度。使用前需配置有效的 Qwen API Key；分享页面由分享者配置。"
        )
        st.caption(
            "支持最长 60 秒的视频：约每秒 2 帧初筛，疑似片段约每秒 4 帧复核，单次最多 18 次请求。"
            "完成后可查看疑似事件、判断依据和证据截图。采样分析可能遗漏短暂动作，结果需人工复核。"
        )
        valid = path is not None
        if valid:
            try:
                video_plan(path)
            except QwenError as error:
                valid = False
                st.warning(str(error))
        else:
            st.caption("请先上传有效的短视频。")
        if st.button("使用 Qwen 检验", key="run_qwen", disabled=not valid, type="primary"):
            st.session_state['qwen_result'] = None
            bar = st.progress(0.0, text="准备视频画面……")
            try:
                result = run_qwen_video(path, requirement, progress=lambda fraction, text: bar.progress(fraction,text=text))
                st.session_state['qwen_result'] = result
                bar.progress(1.0,text="Qwen 检验完成")
            except (QwenError, CredentialStoreError) as error:
                st.error(str(error))
            except Exception:
                st.error("视频检验未完成，请检查本地视频编码及输出目录后重试；本次没有生成有效结论。")
    except QwenError as error:
        st.error(str(error))


def render_qwen_result():
    result = st.session_state.get('qwen_result')
    if not result:
        return
    with st.container(border=True):
        st.subheader("Qwen 检验结果")
        st.write("检验需求：", result['requirement'])
        st.caption(f"来源：{result['source']} · 模型：{result['model']} · 耗时：{result['elapsed_seconds']:.1f}秒")
        if result['cached']:
            st.info("已复用相同视频、需求和模型的检验结果，本次未重复调用 API。")
        else:
            st.caption(f"本次请求 {result['requests']} 次；疑似事件 {len(result['events'])} 个。")
        if any(window['status']=='uncertain' for window in result['windows']):
            st.info("部分片段无法判断，请查看原因或使用更清晰的画面。")
        elif not result['events']:
            st.info("采样范围内未发现满足需求的事件。")
        with VideoSource.open(result['result_video_path']) as source:
            meta = source.metadata()
        video_width = max(1, int(min(meta.width, 640, 480*meta.width/meta.height)))
        with st.container(horizontal=True, horizontal_alignment="center"):
            st.video(result['result_video_path'], width=video_width)
        rows = [{"区间（秒）": f"{w['start']:.2f}–{w['end']:.2f}", "结果": STATUS_NAMES[w['status']],
                 "依据": w['reason'], "局限": w['limitations'], "已复核": w['reviewed']} for w in result['windows']]
        st.dataframe(rows, hide_index=True)
        if result['events']:
            st.dataframe([{"事件":e['event_id'],"状态":"疑似", "首个证据（秒）":round(e['first_evidence_seconds'],3),
                           "最后证据（秒）":round(e['last_evidence_seconds'],3),"依据":e['reason']} for e in result['events']], hide_index=True)
        evidence = result['evidence']
        if evidence:
            st.subheader("证据截图")
            # Keep long reports compact; original evidence is included in the ZIP.
            page = st.selectbox("证据页", list(range(1, (len(evidence)+11)//12+1)), key="qwen_evidence_page")
            columns = st.columns(3)
            for index, frame in enumerate(evidence[(page-1)*12:page*12]):
                with Image.open(frame['path']) as picture:
                    width, height = picture.size
                columns[index%3].image(frame['path'],width=max(1,int(min(width,320,360*width/height))),caption=f"{frame['timestamp']:.2f}秒 · 帧 {frame['frame_id']}")
        folder = Path(result['run_dir'])
        with st.container(horizontal=True):
            for filename, label, mime in [('result.mp4','下载 Qwen 标注视频','video/mp4'),('report.json','下载 Qwen 报告','application/json'),('windows.csv','下载 Qwen 结果 CSV','text/csv')]:
                st.download_button(label,(folder/filename).read_bytes(),file_name=filename,mime=mime,key='qwen_'+filename)
            archive = io.BytesIO()
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
                output.write(folder/'report.json','report.json')
                for frame in evidence:
                    output.write(frame['path'],'evidence/'+Path(frame['path']).name)
            st.download_button("下载证据截图包",archive.getvalue(),file_name='qwen-evidence.zip',mime='application/zip',key='qwen_evidence_zip')

"""Inspect all rule states, including those omitted from the compact overlay."""

from bisect import bisect_right
import json
from pathlib import Path

import streamlit as st


@st.cache_data(show_spinner=False, max_entries=4)
def load_progress(path: str):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def render_progress_view(run_dir: Path) -> None:
    path = run_dir / "rule_progress.jsonl"
    if not path.is_file():
        return
    with st.expander("规则过程与未报警原因", expanded=False):
        entries = load_progress(str(path))
        if not entries:
            st.info("没有规则进度记录。")
            return
        timestamps = [entry["timestamp_seconds"] for entry in entries]
        timestamp = st.slider("查看视频时刻（秒）",0.0,max(0.01,float(timestamps[-1])),0.0,step=0.01,key=f"progress_{run_dir.name}")
        entry = entries[max(0,bisect_right(timestamps,timestamp)-1)]
        st.caption(f"第 {entry['frame_id']} 帧 · {entry['timestamp_seconds']:.2f} 秒。下表显示该时刻所有规则；视频底部最多显示三条。")
        st.dataframe([{"规则":p["label"],"目标 ID":p["track_id"],"进度":f"{p['current']:.2f}/{p['required']:g}{p['unit']}" if p["unit"] else "—","原因":p["reason"]} for p in entry["rules"]],hide_index=True)

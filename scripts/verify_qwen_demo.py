"""Exercise requirement routing and Qwen inspection on the downloaded head-turn clip."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.requirement_guard import RequirementRejected
from src.pipeline.qwen_pipeline import run_qwen_video, video_plan
from src.security.credential_store import load_api_key
from src.ui.app_service import resolve_scene_with_status


def main():
    video = ROOT/'assets/demo_videos/head_turn_control.mp4'
    requirement = '检测视频中的人员是否出现明显转头动作，指出对应画面。'
    meta, _ = video_plan(video)
    try:
        resolve_scene_with_status(mode='deepseek',requirement=requirement,api_key=load_api_key(),
                                  video_fps=meta.fps,video_size=(meta.width,meta.height))
    except RequirementRejected as decision:
        if not decision.vlm_eligible:
            raise RuntimeError('Requirement did not qualify for visual fallback: '+decision.status) from None
        print('DeepSeek route: unsupported local, eligible for Qwen.',flush=True)
    else:
        raise RuntimeError('Unexpected local route for head-turn requirement')
    report = run_qwen_video(video,requirement,progress=lambda value,text: print(f'{value:.0%} {text}',flush=True))
    summary = {key:report[key] for key in ('model','requests','cached','status','run_dir','elapsed_seconds')}
    summary['window_statuses'] = [w['status'] for w in report['windows']]
    summary['events'] = len(report['events'])
    print(json.dumps(summary,ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()

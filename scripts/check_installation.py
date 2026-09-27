"""Verify local CPU model inference and video codecs without any API requests."""

import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cv2
    import numpy as np
    import torch
    from scripts.download_models import load_manifest, model_path, verified
    from src.pipeline.analysis_pipeline import AnalysisPipeline
    from src.video.browser_video import make_browser_playable
    from src.video.video_source import VideoSource
    from src.video.video_writer import VideoWriter

    for item in load_manifest():
        if not verified(model_path(item), item):
            raise RuntimeError(f"Missing or invalid model: {item['path']}; run scripts/download_models.py")
    videos = ["border_demo.avi", "fire_demo.avi", "on_duty_demo.mp4", "drowsiness_demo.mp4",
              "multi_person_clear_1080p.mp4", "multi_face_staggered_composite.mp4"]
    for name in videos:
        with VideoSource.open(ROOT / "assets/demo_videos" / name) as source:
            if next(iter(source), None) is None:
                raise RuntimeError(f"No decodable frames in {name}")
    registry = AnalysisPipeline.from_model_config(ROOT / "config/models.yaml").model_registry
    results = []
    for model_id, video in [("yolo_general", "border_demo.avi"), ("fire_smoke", "fire_demo.avi"),
                            ("face_landmarker", "multi_person_clear_1080p.mp4")]:
        with VideoSource.open(ROOT / "assets/demo_videos" / video) as source:
            frame_id, timestamp, frame = next(iter(source))
        model = registry.create(model_id)
        try:
            if model_id == "face_landmarker":
                model.predict(frame, frame_id, timestamp_seconds=timestamp)
                count = sum(f.valid for f in model.observations)
                if count < 2:
                    raise RuntimeError("Multi-face sample did not yield two valid faces")
            else:
                count = len(model.predict(frame, frame_id))
                if count == 0:
                    raise RuntimeError(f"No detections in fixed example for {model_id}")
            results.append({"model": model_id, "detections": count, "ok": True})
            print(f"OK: {model_id}, detections={count}", flush=True)
        finally:
            model.unload()
    folder = ROOT / "runs/install-check"
    folder.mkdir(parents=True, exist_ok=True)
    raw, playable = folder / "codec-input.mp4", folder / "codec-browser.mp4"
    with VideoWriter.open(raw, fps=10, frame_size=(160, 120)) as writer:
        for _ in range(3):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    make_browser_playable(raw, playable)
    with VideoSource.open(playable) as source:
        if sum(1 for _ in source) != 3:
            raise RuntimeError("Browser video conversion failed")
    report = {"python": sys.version, "platform": platform.platform(), "models": results,
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "example_videos": len(videos), "browser_video": "ok", "api_requests": 0}
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Installation checks passed. No API key is required for offline templates.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Installation check failed: {error}", file=sys.stderr)
        print("See docs/11_fresh_windows_setup.md for dependencies, DLL and network troubleshooting.", file=sys.stderr)
        raise SystemExit(1)

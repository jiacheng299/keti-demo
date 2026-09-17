"""Compare a fire/smoke model on fixed positive and negative videos."""

import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the reproducible fire-model validation interface."""
    parser = argparse.ArgumentParser(description="验证火焰烟雾模型的 CPU 基线")
    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--positive-video", required=True, help="火焰或烟雾正样本")
    parser.add_argument("--negative-video", required=True, help="无火焰烟雾负样本")
    parser.add_argument("--imgsz", type=int, default=416, help="模型输入尺寸")
    parser.add_argument("--confidence", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--positive-max-frames", type=int, default=0)
    parser.add_argument("--negative-max-frames", type=int, default=100)
    parser.add_argument("--output-dir", default="runs/validation/fire")
    return parser


def count_classes(
    class_ids: Sequence[int], class_names: Mapping[int, str]
) -> dict[str, int]:
    """Count detections using the class labels embedded in model metadata."""
    return dict(Counter(class_names[int(class_id)] for class_id in class_ids))


def validate_clip(
    model: object,
    video_path: str | Path,
    output_path: str | Path,
    imgsz: int,
    confidence: float,
    max_frames: int,
) -> dict[str, object]:
    """Run one video through the model on CPU and save annotated evidence."""
    import cv2

    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频不存在：{video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法读取视频：{video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"视频元数据无效：{video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"无法创建结果视频：{output_path}")

    names = model.names
    class_totals = {name: 0 for name in names.values()}
    frame_count = 0
    frames_with_detection = 0
    inference_seconds = 0.0
    started_at = time.perf_counter()

    try:
        while max_frames <= 0 or frame_count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            inference_started_at = time.perf_counter()
            result = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=confidence,
                device="cpu",
                verbose=False,
            )[0]
            inference_seconds += time.perf_counter() - inference_started_at

            if result.boxes is not None and len(result.boxes) > 0:
                frames_with_detection += 1
                frame_counts = count_classes(
                    result.boxes.cls.cpu().tolist(), result.names
                )
                for class_name, count in frame_counts.items():
                    class_totals[class_name] = class_totals.get(class_name, 0) + count

            writer.write(result.plot())
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    elapsed_seconds = time.perf_counter() - started_at
    if frame_count == 0:
        raise ValueError(f"视频中没有可处理的视频帧：{video_path}")

    return {
        "source": str(video_path.resolve()),
        "frames": frame_count,
        "frames_with_detection": frames_with_detection,
        "class_counts": class_totals,
        "average_inference_ms": round(inference_seconds * 1000 / frame_count, 3),
        "effective_fps": round(frame_count / elapsed_seconds, 3),
        "output": str(output_path.resolve()),
    }


def main() -> None:
    """Validate one model on fixed positive and negative clips."""
    from ultralytics import YOLO

    args = build_parser().parse_args()
    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"权重不存在：{weights_path}")

    model = YOLO(str(weights_path))
    supported_classes = set(model.names.values())
    if not {"fire", "smoke"}.issubset(supported_classes):
        raise ValueError(f"模型类别不包含 fire 和 smoke：{model.names}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "weights": str(weights_path.resolve()),
        "device": "cpu",
        "classes": model.names,
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "positive": validate_clip(
            model,
            args.positive_video,
            output_dir / "positive_annotated.mp4",
            args.imgsz,
            args.confidence,
            args.positive_max_frames,
        ),
        "negative": validate_clip(
            model,
            args.negative_video,
            output_dir / "negative_annotated.mp4",
            args.imgsz,
            args.confidence,
            args.negative_max_frames,
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

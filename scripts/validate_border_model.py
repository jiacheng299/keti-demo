"""CPU baseline validation helpers for the border detector."""

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path


BORDER_CLASS_IDS = [0, 2, 3, 5, 7]


def build_parser() -> argparse.ArgumentParser:
    """Build the reproducible command-line interface for baseline runs."""
    parser = argparse.ArgumentParser(description="验证边防通用检测模型的 CPU 基线")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--imgsz", type=int, default=640, help="模型输入尺寸")
    parser.add_argument(
        "--output",
        default="runs/validation/border_sample.mp4",
        help="标注结果视频路径",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最多处理帧数，0 表示处理完整视频",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="置信度阈值")
    return parser


def normalize_detections(
    boxes: Sequence[Sequence[float]],
    confidences: Sequence[float],
    class_ids: Sequence[int],
    class_names: Mapping[int, str],
    frame_id: int,
) -> list[dict[str, object]]:
    """Convert detector arrays into the demo's temporary normalized format."""
    return [
        {
            "frame_id": frame_id,
            "class_id": int(class_id),
            "class_name": class_names[int(class_id)],
            "confidence": float(confidence),
            "bbox_xyxy": [float(value) for value in box],
        }
        for box, confidence, class_id in zip(
            boxes, confidences, class_ids, strict=True
        )
    ]


def run_validation(
    video_path: str | Path,
    weights_path: str | Path,
    imgsz: int,
    output_path: str | Path,
    max_frames: int = 0,
    confidence: float = 0.25,
) -> dict[str, object]:
    """Run a CPU-only detector baseline and write an annotated video."""
    import cv2
    from ultralytics import YOLO

    video_path = Path(video_path)
    weights_path = Path(weights_path)
    output_path = Path(output_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频不存在：{video_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"权重不存在：{weights_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法读取视频：{video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("视频元数据无效，无法创建结果视频")

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

    model = YOLO(str(weights_path))
    frame_count = 0
    detection_count = 0
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
                classes=BORDER_CLASS_IDS,
                device="cpu",
                verbose=False,
            )[0]
            inference_seconds += time.perf_counter() - inference_started_at

            if result.boxes is not None:
                detections = normalize_detections(
                    boxes=result.boxes.xyxy.cpu().tolist(),
                    confidences=result.boxes.conf.cpu().tolist(),
                    class_ids=result.boxes.cls.cpu().tolist(),
                    class_names=result.names,
                    frame_id=frame_count,
                )
                detection_count += len(detections)

            writer.write(result.plot())
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    elapsed_seconds = time.perf_counter() - started_at
    if frame_count == 0:
        raise ValueError("视频中没有可处理的视频帧")

    return {
        "device": "cpu",
        "frames": frame_count,
        "detections": detection_count,
        "imgsz": imgsz,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "average_inference_ms": round(inference_seconds * 1000 / frame_count, 3),
        "effective_fps": round(frame_count / elapsed_seconds, 3),
        "output": str(output_path.resolve()),
    }


def main() -> None:
    """Parse arguments, run the baseline, and print reproducible metrics."""
    args = build_parser().parse_args()
    metrics = run_validation(
        video_path=args.video,
        weights_path=args.weights,
        imgsz=args.imgsz,
        output_path=args.output,
        max_frames=args.max_frames,
        confidence=args.confidence,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

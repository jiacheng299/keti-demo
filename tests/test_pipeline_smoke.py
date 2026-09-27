import csv
import json
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

from src.models.model_registry import ModelRegistry
from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.pipeline.contracts import PipelineProgress
from src.schemas.detection import Detection
from src.schemas.scene_spec import SceneSpec
from src.video.frame_sampler import FrameSampler
from src.visualization.scene_overlay import SceneOverlay


FRAME_SIZE = (64, 48)
FPS = 10.0


def create_synthetic_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        FPS,
        FRAME_SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError("test environment cannot create an MJPG video")
    try:
        for index in range(frame_count):
            writer.write(
                np.full(
                    (FRAME_SIZE[1], FRAME_SIZE[0], 3),
                    fill_value=index * 40,
                    dtype=np.uint8,
                )
            )
    finally:
        writer.release()


def count_video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        count = 0
        while True:
            readable, _frame = capture.read()
            if not readable:
                return count
            count += 1
    finally:
        capture.release()


class FakeAdapter:
    def __init__(self, detection_factory):
        self.detection_factory = detection_factory
        self.loaded = False
        self.unloaded = False
        self.frame_ids = []

    def load(self) -> None:
        self.loaded = True

    def predict(self, _frame, frame_id=0):
        if not self.loaded:
            raise RuntimeError("adapter must be loaded before prediction")
        self.frame_ids.append(frame_id)
        return self.detection_factory(frame_id)

    def unload(self) -> None:
        self.unloaded = True


def fire_detections(frame_id: int) -> list[Detection]:
    return [
        Detection(
            frame_id=frame_id,
            class_name="fire",
            confidence=0.91,
            bbox_xyxy=(10, 10, 30, 30),
        )
    ]


def make_registry(adapter_name: str, adapter: FakeAdapter) -> ModelRegistry:
    registry = ModelRegistry.from_config("config/models.yaml")
    registry.register_factory(adapter_name, lambda _definition: adapter)
    return registry


def fire_scene() -> SceneSpec:
    return SceneSpec.model_validate(
        {
            "scene_type": "fire",
            "model_id": "fire_smoke",
            "targets": ["fire"],
            "rules": [
                {
                    "type": "consecutive_frames",
                    "params": {
                        "rule_id": "fire_sequence",
                        "frames": 2,
                        "max_gap_frames": 0,
                    },
                }
            ],
            "alert": {"enabled": True, "cooldown_seconds": 0},
        }
    )


def border_detections(frame_id: int) -> list[Detection]:
    center_x = 20 if frame_id == 0 else 35
    return [
        Detection(
            frame_id=frame_id,
            class_name="person",
            confidence=0.9,
            bbox_xyxy=(center_x - 5, 15, center_x + 5, 25),
        )
    ]


def border_scene() -> SceneSpec:
    return SceneSpec.model_validate(
        {
            "scene_type": "border",
            "model_id": "yolo_general",
            "targets": ["person"],
            "rules": [
                {
                    "type": "enter_region",
                    "params": {
                        "region_id": "restricted_zone",
                        "polygon": [[30, 5], [60, 5], [60, 40], [30, 40]],
                    },
                }
            ],
            "alert": {"enabled": True, "cooldown_seconds": 0},
        }
    )


def test_fire_pipeline_writes_video_events_and_exports(tmp_path):
    video_path = tmp_path / "input.avi"
    run_dir = tmp_path / "run-fire"
    create_synthetic_video(video_path)
    adapter = FakeAdapter(fire_detections)
    progress = []
    pipeline = AnalysisPipeline(make_registry("fire", adapter))

    summary = pipeline.run(video_path, fire_scene(), run_dir, progress.append)

    assert adapter.loaded is True
    assert adapter.unloaded is True
    assert adapter.frame_ids == [0, 1, 2]
    assert [event.alert_status for event in summary.events] == [
        "suspected",
        "confirmed",
    ]
    assert summary.frames_read == 3
    assert summary.processed_frames == 3
    assert summary.detection_count == 3
    assert summary.stopped is False
    assert progress[-1].fraction == pytest.approx(1.0)
    assert count_video_frames(summary.result_video_path) == 3

    json_rows = json.loads((run_dir / "events.json").read_text("utf-8"))
    with (run_dir / "events.csv").open(encoding="utf-8-sig", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    exported_summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
    assert len(json_rows) == len(csv_rows) == exported_summary["event_count"] == 2
    assert exported_summary["metrics"]["processed_frames"] == 3

    snapshot = np.fromfile(run_dir / "snapshots" / "evt-000002.jpg", dtype=np.uint8)
    assert cv2.imdecode(snapshot, cv2.IMREAD_COLOR) is not None


def test_border_pipeline_tracks_person_and_emits_enter_event(tmp_path):
    video_path = tmp_path / "input.avi"
    run_dir = tmp_path / "run-border"
    create_synthetic_video(video_path)
    adapter = FakeAdapter(border_detections)
    pipeline = AnalysisPipeline(make_registry("yolo", adapter))

    summary = pipeline.run(video_path, border_scene(), run_dir)

    assert len(summary.events) == 1
    assert summary.events[0].event_type == "enter_region"
    assert summary.events[0].track_id == 0
    assert summary.events[0].snapshot_path == "snapshots/evt-000001.jpg"
    snapshot = np.fromfile(run_dir / summary.events[0].snapshot_path, dtype=np.uint8)
    assert cv2.imdecode(snapshot, cv2.IMREAD_COLOR) is not None


def test_pipeline_samples_inference_but_preserves_result_video_length(tmp_path):
    video_path = tmp_path / "input.avi"
    run_dir = tmp_path / "run-sampled"
    create_synthetic_video(video_path)
    adapter = FakeAdapter(fire_detections)
    pipeline = AnalysisPipeline(
        make_registry("fire", adapter),
        frame_sampler=FrameSampler(every_n_frames=2),
    )

    summary = pipeline.run(video_path, fire_scene(), run_dir)

    assert adapter.frame_ids == [0, 2]
    assert summary.frames_read == 3
    assert summary.processed_frames == 2
    assert count_video_frames(summary.result_video_path) == 3


def test_event_notice_survives_frames_without_inference_and_is_saved_in_snapshot(tmp_path, monkeypatch):
    video_path = tmp_path / "input.avi"
    create_synthetic_video(video_path)
    drawn_events = []
    original_draw = SceneOverlay.draw

    def capture_draw(self, frame, detections):
        drawn_events.append([event.event_type for event in self.active_events])
        return original_draw(self, frame, detections)

    monkeypatch.setattr(SceneOverlay, "draw", capture_draw)
    adapter = FakeAdapter(fire_detections)
    summary = AnalysisPipeline(
        make_registry("fire", adapter), frame_sampler=FrameSampler(every_n_frames=2),
    ).run(video_path, fire_scene(), tmp_path / "run")
    assert drawn_events == [["fire_suspected"], ["fire_suspected"], ["fire_confirmed"]]
    snapshot = cv2.imdecode(
        np.frombuffer((summary.run_dir / summary.events[-1].snapshot_path).read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    # The banner background is dark, unlike the gray source frame at confirmation.
    assert np.mean(snapshot[3:8, 40:55]) < 60


def test_pipeline_stop_exports_partial_results_and_unloads_model(tmp_path):
    video_path = tmp_path / "input.avi"
    run_dir = tmp_path / "run-stopped"
    create_synthetic_video(video_path)
    adapter = FakeAdapter(fire_detections)
    pipeline = AnalysisPipeline(make_registry("fire", adapter))
    stop_event = Event()

    def stop_after_first_frame(progress):
        if progress.frames_read == 1:
            stop_event.set()

    summary = pipeline.run(
        video_path,
        fire_scene(),
        run_dir,
        stop_after_first_frame,
        stop_event,
    )

    assert summary.stopped is True
    assert summary.frames_read == 1
    assert adapter.unloaded is True
    assert count_video_frames(summary.result_video_path) == 1
    exported = json.loads((run_dir / "summary.json").read_text("utf-8"))
    assert exported["metrics"]["stopped"] is True
    assert exported["event_count"] == 1


def test_pipeline_unloads_model_and_closes_videos_after_processing_error(tmp_path):
    video_path = tmp_path / "input.avi"
    run_dir = tmp_path / "run-error"
    create_synthetic_video(video_path)

    def fail_on_second_frame(frame_id):
        if frame_id == 1:
            raise RuntimeError("simulated inference failure")
        return fire_detections(frame_id)

    adapter = FakeAdapter(fail_on_second_frame)
    pipeline = AnalysisPipeline(make_registry("fire", adapter))

    with pytest.raises(RuntimeError, match="simulated inference failure"):
        pipeline.run(video_path, fire_scene(), run_dir)

    assert adapter.unloaded is True
    video_path.unlink()
    (run_dir / "result.mp4").unlink()


def test_default_pipeline_registers_both_local_adapter_types():
    pipeline = AnalysisPipeline.from_model_config("config/models.yaml")

    assert type(pipeline.model_registry.create("yolo_general")).__name__ == "YoloAdapter"
    assert type(pipeline.model_registry.create("fire_smoke")).__name__ == "FireAdapter"


def test_progress_fraction_is_zero_when_total_frame_count_is_unknown():
    progress = PipelineProgress(
        frames_read=0,
        total_frames=0,
        processed_frames=0,
        event_count=0,
    )

    assert progress.fraction == 0.0

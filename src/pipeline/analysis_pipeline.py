"""Orchestrate one validated local-video analysis run."""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Protocol

from src.events import EventExporter, EventManager
from src.models.fire_adapter import FireAdapter
from src.models.model_registry import ModelRegistry
from src.models.yolo_adapter import YoloAdapter
from src.rules.base_rule import FrameState
from src.rules.rule_engine import RuleEngine
from src.schemas.event import Event
from src.schemas.scene_spec import SceneSpec
from src.tracking.tracker import Tracker
from src.video.frame_sampler import FrameSampler
from src.video.video_source import VideoSource
from src.video.video_writer import VideoWriter
from src.visualization.annotator import Annotator

from .contracts import PipelineProgress, RunSummary


class StopSignal(Protocol):
    def is_set(self) -> bool: ...


ProgressCallback = Callable[[PipelineProgress], None]


class AnalysisPipeline:
    """Connect existing processing modules without owning their algorithms."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        *,
        frame_sampler: FrameSampler | None = None,
    ) -> None:
        self.model_registry = model_registry
        self.frame_sampler = frame_sampler or FrameSampler()

    @classmethod
    def from_model_config(
        cls,
        config_path: str | Path,
        *,
        frame_sampler: FrameSampler | None = None,
    ) -> "AnalysisPipeline":
        """Build the standard CPU pipeline from the trusted model registry."""
        registry = ModelRegistry.from_config(config_path)
        registry.register_factory("yolo", YoloAdapter)
        registry.register_factory("fire", FireAdapter)
        return cls(registry, frame_sampler=frame_sampler)

    def run(
        self,
        video_path: str | Path,
        scene_spec: SceneSpec,
        run_dir: str | Path,
        progress_callback: ProgressCallback | None = None,
        stop_event: StopSignal | None = None,
    ) -> RunSummary:
        """Analyze one file, export evidence, and return a UI-friendly summary."""
        output_dir = Path(run_dir)
        result_video_path = output_dir / "result.mp4"
        adapter = self.model_registry.create(scene_spec.model_id)
        tracker = Tracker()
        rule_engine = RuleEngine()
        event_manager = EventManager(
            output_dir,
            scene_spec.scene_type,
            scene_spec.model_id,
        )
        events: list[Event] = []
        frames_read = 0
        processed_frames = 0
        detection_count = 0
        stopped = False
        started_at = perf_counter()

        try:
            adapter.load()
            with VideoSource.open(video_path) as source:
                metadata = source.metadata()
                with VideoWriter.open(
                    result_video_path,
                    fps=metadata.fps,
                    frame_size=(metadata.width, metadata.height),
                ) as writer:
                    for frame_id, timestamp_seconds, frame in source:
                        if stop_event is not None and stop_event.is_set():
                            stopped = True
                            break

                        annotated = frame
                        if self.frame_sampler.should_process(frame_id):
                            detections = adapter.predict(frame, frame_id)
                            processed_frames += 1
                            detection_count += len(detections)
                            if scene_spec.scene_type == "border":
                                detections = tracker.update(
                                    detections,
                                    timestamp_seconds,
                                )

                            candidates = rule_engine.evaluate(
                                scene_spec,
                                detections,
                                FrameState(
                                    frame_id=frame_id,
                                    timestamp_seconds=timestamp_seconds,
                                ),
                            )
                            annotated = Annotator.draw(frame, detections)
                            for candidate in candidates:
                                event = event_manager.add(candidate, annotated)
                                if event is not None:
                                    events.append(event)

                        writer.write(annotated)
                        frames_read += 1
                        if progress_callback is not None:
                            progress_callback(
                                PipelineProgress(
                                    frames_read=frames_read,
                                    total_frames=metadata.frame_count,
                                    processed_frames=processed_frames,
                                    event_count=len(events),
                                )
                            )
        finally:
            adapter.unload()

        elapsed_seconds = perf_counter() - started_at
        metrics = {
            "frames_read": frames_read,
            "processed_frames": processed_frames,
            "detection_count": detection_count,
            "event_count": len(events),
            "elapsed_seconds": elapsed_seconds,
            "processing_fps": (
                processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0
            ),
            "stopped": stopped,
        }
        EventExporter().export(output_dir, scene_spec, events, metrics)
        return RunSummary(
            run_dir=output_dir,
            result_video_path=result_video_path,
            events=tuple(events),
            frames_read=frames_read,
            processed_frames=processed_frames,
            detection_count=detection_count,
            stopped=stopped,
            elapsed_seconds=elapsed_seconds,
        )

"""Repeat real demo scenarios and validate every exported artifact."""

from dataclasses import dataclass
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.ui.app_service import MODEL_CONFIG_PATH, PROJECT_ROOT, resolve_scene_spec
from src.video.browser_video import make_browser_playable


REQUIRED_ARTIFACTS = (
    "config.json",
    "events.json",
    "events.csv",
    "summary.json",
    "result.mp4",
)


class AcceptanceValidationError(RuntimeError):
    """Raised when one completed run does not meet the delivery contract."""


@dataclass(frozen=True)
class AcceptanceScenario:
    name: str
    video_path: Path
    template_id: str
    expected_min_events: int = 1


def validate_run_artifacts(
    run_dir: str | Path,
    expected_min_events: int,
) -> dict[str, int | str]:
    """Cross-check structured outputs, snapshots, and browser video."""
    root = Path(run_dir).resolve()
    for filename in REQUIRED_ARTIFACTS:
        if not (root / filename).is_file():
            raise AcceptanceValidationError(f"missing artifact: {filename}")

    try:
        events = json.loads((root / "events.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        with (root / "events.csv").open(encoding="utf-8-sig", newline="") as file:
            csv_events = list(csv.DictReader(file))
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as error:
        raise AcceptanceValidationError("structured artifact is unreadable") from error

    if not isinstance(events, list) or not isinstance(summary, dict):
        raise AcceptanceValidationError("structured artifact has an invalid shape")
    summary_count = summary.get("event_count")
    if len(events) != len(csv_events) or summary_count != len(events):
        raise AcceptanceValidationError("event counts do not match")
    if len(events) < expected_min_events:
        raise AcceptanceValidationError(
            f"event count {len(events)} is below expected minimum {expected_min_events}"
        )

    snapshot_root = (root / "snapshots").resolve()
    snapshot_count = 0
    for event in events:
        if not isinstance(event, dict):
            raise AcceptanceValidationError("event row has an invalid shape")
        snapshot_path = event.get("snapshot_path")
        if snapshot_path is None:
            continue
        if not isinstance(snapshot_path, str):
            raise AcceptanceValidationError("snapshot path has an invalid type")
        candidate = (root / snapshot_path).resolve()
        try:
            candidate.relative_to(snapshot_root)
        except ValueError as error:
            raise AcceptanceValidationError("snapshot path escapes run directory") from error
        try:
            encoded = np.frombuffer(candidate.read_bytes(), dtype=np.uint8)
        except OSError as error:
            raise AcceptanceValidationError(
                f"snapshot is unreadable: {snapshot_path}"
            ) from error
        if cv2.imdecode(encoded, cv2.IMREAD_COLOR) is None:
            raise AcceptanceValidationError(f"snapshot is unreadable: {snapshot_path}")
        snapshot_count += 1

    video_path = root / "result.mp4"
    capture = cv2.VideoCapture(str(video_path))
    try:
        readable, _frame = capture.read()
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not readable or frame_count <= 0:
        raise AcceptanceValidationError("result video is unreadable")
    if b"avc1" not in video_path.read_bytes():
        raise AcceptanceValidationError("result video is not H.264")

    return {
        "event_count": len(events),
        "csv_event_count": len(csv_events),
        "snapshot_count": snapshot_count,
        "video_frame_count": frame_count,
        "browser_codec": "h264",
    }


def build_default_scenarios(project_root: str | Path = PROJECT_ROOT) -> tuple[AcceptanceScenario, ...]:
    root = Path(project_root)
    return (
        AcceptanceScenario(
            name="border",
            video_path=root / "assets" / "demo_videos" / "border_demo.avi",
            template_id="border_person_intrusion",
        ),
        AcceptanceScenario(
            name="fire",
            video_path=root / "assets" / "demo_videos" / "fire_demo.avi",
            template_id="fire_detection",
        ),
    )


def run_acceptance(
    scenarios: tuple[AcceptanceScenario, ...],
    *,
    repeat: int,
    output_root: str | Path,
    backup_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run every offline scenario repeatedly and write a JSON evidence report."""
    if repeat < 1:
        raise ValueError("repeat must be at least one")
    if not scenarios:
        raise ValueError("at least one acceptance scenario is required")

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report_rows: list[dict[str, Any]] = []
    latest_runs: dict[str, Path] = {}

    for iteration in range(1, repeat + 1):
        for scenario in scenarios:
            if not scenario.video_path.is_file():
                raise FileNotFoundError(f"acceptance video does not exist: {scenario.video_path}")
            print(f"[{scenario.name} {iteration}/{repeat}] starting", flush=True)
            scene = resolve_scene_spec(
                mode="offline",
                requirement="",
                template_id=scenario.template_id,
            )
            pipeline = AnalysisPipeline.from_model_config(MODEL_CONFIG_PATH)
            run_dir = root / f"{scenario.name}-{iteration}"
            wall_started = perf_counter()
            summary = pipeline.run(scenario.video_path, scene, run_dir)
            make_browser_playable(summary.result_video_path, summary.result_video_path)
            evidence = validate_run_artifacts(run_dir, scenario.expected_min_events)
            wall_seconds = perf_counter() - wall_started
            row = {
                "scenario": scenario.name,
                "iteration": iteration,
                "template_id": scenario.template_id,
                "model_id": scene.model_id,
                "video_path": str(scenario.video_path.resolve()),
                "run_dir": str(run_dir),
                "pipeline_seconds": round(summary.elapsed_seconds, 3),
                "wall_seconds": round(wall_seconds, 3),
                **evidence,
            }
            report_rows.append(row)
            latest_runs[scenario.name] = run_dir
            print(
                f"[{scenario.name} {iteration}/{repeat}] passed in {wall_seconds:.2f}s",
                flush=True,
            )

    report = {
        "status": "passed",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "offline_templates": True,
        "deepseek_network_calls": 0,
        "repeat": repeat,
        "run_count": len(report_rows),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "runs": report_rows,
    }
    _write_json_atomically(root / "acceptance_report.json", report)

    if backup_root is not None:
        backup = Path(backup_root).resolve() / root.name
        for scenario_name, run_dir in latest_runs.items():
            shutil.copytree(run_dir, backup / scenario_name)
        report["backup_root"] = str(backup)
        _write_json_atomically(root / "acceptance_report.json", report)
    return report


def _write_json_atomically(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

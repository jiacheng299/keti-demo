"""Pure helpers used by the Streamlit entry point."""

from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from typing import Protocol
from uuid import uuid4

from src.llm.deepseek_client import DeepSeekClient
from src.llm.scene_parser import SceneParser
from src.models.model_registry import ModelRegistry
from src.schemas.scene_spec import SceneSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
TEMPLATE_DIR = PROJECT_ROOT / "config" / "scenes"
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".ogv", ".mkv"}
TEMPLATE_OPTIONS: Mapping[str, str] = {
    "边防人员禁区": "border_person_intrusion",
    "边防车辆禁区": "border_vehicle_intrusion",
    "边防人员滞留": "border_dwell",
    "火灾检测": "fire_detection",
}
EVENT_TYPE_LABELS = {
    "enter_region": "进入禁区",
    "leave_region": "离开区域",
    "dwell": "区域滞留",
    "fire_suspected": "疑似火灾",
    "fire_confirmed": "确认火灾",
}
STATUS_LABELS = {"suspected": "疑似", "confirmed": "确认"}


class UnsupportedVideoTypeError(ValueError):
    """Raised before persisting an upload with an unapproved extension."""


class CompletionClient(Protocol):
    def complete(self, messages): ...


def resolve_scene_spec(
    *,
    mode: str,
    requirement: str,
    template_id: str,
    client: CompletionClient | None = None,
) -> SceneSpec:
    """Resolve one validated template or DeepSeek-generated configuration."""
    registry = ModelRegistry.from_config(MODEL_CONFIG_PATH)
    parser = SceneParser(
        client or DeepSeekClient(),
        registry,
        template_dir=TEMPLATE_DIR,
    )
    if mode == "offline":
        return parser.load_template(template_id)
    if mode == "deepseek":
        return parser.parse_or_template(requirement, template_id)
    raise ValueError(f"unsupported scene configuration mode: {mode}")


def save_uploaded_video(
    content: bytes,
    original_name: str,
    upload_dir: str | Path,
) -> Path:
    """Persist upload bytes under a content-derived filename inside one directory."""
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise UnsupportedVideoTypeError(f"unsupported video type: {suffix or '<none>'}")
    if not content:
        raise ValueError("uploaded video is empty")

    destination_dir = Path(upload_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{sha256(content).hexdigest()[:16]}{suffix}"
    if destination.exists():
        return destination

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination_dir,
            prefix=".upload-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def create_run_dir(root: str | Path = PROJECT_ROOT / "runs") -> Path:
    """Return a collision-resistant directory path for one analysis run."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(root) / f"{timestamp}-{uuid4().hex[:8]}"


def event_rows_for_display(events: list[dict]) -> list[dict]:
    """Select and localize the event fields useful during a demo."""
    return [
        {
            "事件编号": event["event_id"],
            "事件类型": EVENT_TYPE_LABELS.get(event["event_type"], event["event_type"]),
            "目标类别": event["target_class"],
            "目标 ID": "-" if event.get("track_id") is None else event["track_id"],
            "视频时间（秒）": f"{event['timestamp_seconds']:.3f}",
            "状态": STATUS_LABELS.get(event["alert_status"], event["alert_status"]),
        }
        for event in events
    ]


def snapshot_paths_for_display(
    run_dir: str | Path,
    events: list[dict],
) -> list[tuple[Path, dict]]:
    """Return existing event screenshots contained by the run snapshot folder."""
    run_root = Path(run_dir).resolve()
    snapshot_root = (run_root / "snapshots").resolve()
    snapshots: list[tuple[Path, dict]] = []
    for event in events:
        relative_path = event.get("snapshot_path")
        if not isinstance(relative_path, str) or not relative_path:
            continue
        candidate = (run_root / relative_path).resolve()
        try:
            candidate.relative_to(snapshot_root)
        except ValueError:
            continue
        if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            snapshots.append((candidate, event))
    return snapshots

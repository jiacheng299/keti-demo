"""Pure helpers used by the Streamlit entry point."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import os
import re
from pathlib import Path
import tempfile
from typing import Protocol
from uuid import uuid4

from src.llm.deepseek_client import DeepSeekAPIError, DeepSeekClient, MissingAPIKeyError
from src.llm.scene_parser import SceneParseError, SceneParser
from src.llm.requirement_guard import RequirementRejected, validate_requirement_text
from src.models.model_registry import ModelRegistry
from src.schemas.scene_spec import SceneSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
TEMPLATE_DIR = PROJECT_ROOT / "config" / "scenes"
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".ogv", ".mkv"}
TEMPLATE_OPTIONS: Mapping[str, str] = {
    "边防禁区": "border_intrusion",
    "边防人员滞留": "border_dwell",
    "火灾检测": "fire_detection",
    "人员在岗监测": "on_duty",
    "打瞌睡监测": "drowsiness",
}
EVENT_TYPE_LABELS = {
    "post_unstaffed": "岗位缺员",
    "post_recovered": "回岗恢复",
    "drowsiness_suspected": "疑似打瞌睡（持续闭眼）",
    "eyes_reopened": "睁眼恢复",
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
    api_key: str | None = None,
    video_fps: float | None = None,
) -> SceneSpec:
    """Resolve one validated template or DeepSeek-generated configuration."""
    return resolve_scene_with_status(
        mode=mode, requirement=requirement, template_id=template_id,
        client=client, api_key=api_key, video_fps=video_fps,
    ).scene


@dataclass(frozen=True)
class SceneResolution:
    scene: SceneSpec
    source: str
    warning: str | None = None
    requirement_summary: str | None = None


def requires_video_fps(requirement: str) -> bool:
    """Fire duration needs the actual video timebase, unlike a dwell rule."""
    return bool(re.search(r"火焰|火灾|火苗|起火|着火|烟雾|浓烟|\bfire\b|\bsmoke\b", requirement, re.IGNORECASE)) and bool(
        re.search(r"秒|分钟|seconds?\b|minutes?\b|\d\s*s\b", requirement, re.IGNORECASE)
    )


def resolve_scene_with_status(
    *, mode: str, requirement: str, template_id: str | None = None,
    client: CompletionClient | None = None, api_key: str | None = None,
    video_fps: float | None = None,
    video_size: tuple[int, int] | None = None,
) -> SceneResolution:
    """Only fall back when the caller explicitly supplies an offline template."""
    registry = ModelRegistry.from_config(MODEL_CONFIG_PATH)
    parser = SceneParser(
        client if client is not None else DeepSeekClient(api_key=api_key),
        registry,
        template_dir=TEMPLATE_DIR,
    )
    if mode == "offline":
        if template_id is None:
            raise ValueError("请选择离线模板。")
        return SceneResolution(parser.load_template(template_id), "离线模板")
    if mode == "deepseek":
        validate_requirement_text(requirement)
        if video_fps is None and requires_video_fps(requirement):
            raise ValueError("请先上传视频，再按实际帧率将火灾持续时间换算为确认帧数。")
        try:
            scene, summary = parser.parse_requirement(requirement, video_fps=video_fps, video_size=video_size)
            return SceneResolution(scene, "DeepSeek", requirement_summary=summary)
        except RequirementRejected:
            # Unsupported intent must never silently become an offline template.
            raise
        except MissingAPIKeyError:
            reason = "未配置 API Key，请在“DeepSeek API 设置”中保存密钥。"
        except DeepSeekAPIError as error:
            reason = {
                400: "请求格式被接口拒绝（HTTP 400）。",
                401: "API Key 无效或已失效（HTTP 401），请检查后重新保存。",
                402: "DeepSeek API 账户余额不足（HTTP 402）。",
                403: "当前 API Key 没有访问权限（HTTP 403）。",
                404: "请求的接口或模型不可用（HTTP 404）。",
                422: "接口不接受当前请求参数（HTTP 422）。",
                429: "请求过于频繁（HTTP 429），请稍后重试。",
                500: "DeepSeek 服务异常（HTTP 500），请稍后重试。",
                503: "DeepSeek 服务繁忙（HTTP 503），请稍后重试。",
            }.get(error.status_code)
            if reason is None:
                reason = {
                    "timeout": "DeepSeek 请求超时（单次等待上限60秒，已重试一次），请稍后重试。",
                    "connection_error": "无法连接 DeepSeek 接口，请检查网络、代理或证书配置后重试。",
                    "invalid_response": "接口返回的内容不是有效的 DeepSeek 响应，可能是代理或服务异常，请稍后重试。",
                    "output_truncated": "DeepSeek 输出达到长度上限，配置未生成完整，请减少区域或规则后重试。",
                    "empty_response": "DeepSeek 返回了空的最终答案，请重新生成场景配置。",
                    "content_filtered": "DeepSeek 未返回可用内容，请调整场景需求后重试。",
                }.get(error.error_code, "DeepSeek 接口调用失败，请稍后重试。")
        except SceneParseError:
            reason = "DeepSeek 返回的配置不符合当前支持的场景和规则，请简化需求后重试。"
        if template_id is None:
            raise ValueError(f"{reason} 请重试，或切换到“离线模板”后选择场景。")
        return SceneResolution(
            parser.load_template(template_id), "离线回退",
            f"{reason} 本次已使用所选离线模板，文字需求未生效。",
        )
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

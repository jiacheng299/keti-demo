"""Resolve duration intent locally against the video's actual timebase."""

import math

from src.schemas.scene_spec import SceneSpec


def resolve_time_rules(scene: SceneSpec, fps: float, every_n_frames: int = 1) -> SceneSpec:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("视频帧率必须为正数")
    result = scene.model_copy(deep=True)
    for rule in result.rules:
        if rule.type == "consecutive_frames" and "seconds" in rule.params:
            seconds = rule.params["seconds"]
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0.1 <= seconds <= 3600:
                raise ValueError("持续时间须在 0.1–3600 秒之间")
            rule.params["frames"] = math.ceil(seconds * fps / every_n_frames) + 1
    return result

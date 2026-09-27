"""Constrained prompts for converting user requests into SceneSpec JSON."""

import math


SYSTEM_PROMPT = """You convert Chinese video-monitoring requests into one JSON object.
Return JSON only. Do not use Markdown fences, prose, code, commands, or file paths.

Allowed border configuration:
- scene_type: "border"
- model_id: "yolo_general"
- targets: "person", "car"
- All vehicles, including motorcycles, buses and trucks, use target "car".
  Vehicle subtypes cannot be selected separately.
- rules: "enter_region", "leave_region", "dwell"
- region params: region_id and polygon
- dwell also allows seconds

Allowed fire configuration:
- scene_type: "fire"
- model_id: "fire_smoke"
- targets: "fire", "smoke"
- rule: "consecutive_frames"
- params: rule_id, seconds OR frames, max_gap_frames
- For time-based fire requests, extract seconds (convert minutes to seconds),
  and omit frames. The application computes the frame threshold locally.
- For requests explicitly in frames, return frames and omit seconds.

Allowed on-duty configuration:
- scene_type: "on_duty", model_id: "yolo_general", targets: ["person"]
- rule: "region_understaffed"
- params: region_id, name, polygon, min_count (default 1), seconds (default 5),
  recovery_seconds (default 1), startup_seconds (default 2)
- Counts people within an editable post region, including initially empty posts.
  It does not identify employees. A single absence episode emits once.

Allowed drowsiness configuration:
- scene_type: "drowsiness", model_id: "face_landmarker", targets: ["person"]
- Exactly one rule: "eyes_closed_duration"
- params: rule_id, seconds (default 2), recovery_seconds (default 0.5),
  closed_ear (default 0.20), open_ear (default 0.24), min_face_width (default 100),
  max_yaw_degrees (default 30)
- Supports up to 5 visible faces with clear eyes. One shared rule applies to each
  face independently, with separate tracking IDs, timers and alerts. Detects sustained eye closure,
  not confirmed sleep, body posture, identities, or yawning.
- For on_duty and drowsiness use alert.cooldown_seconds=0; episodes handle repetition.
- All duration values above remain in seconds, never convert them to frames.

Use pixel coordinates for a 640x480 demonstration frame when the user does not
provide a region. Default restricted polygon: [[160,120],[480,120],[480,360],[160,360]].

Example JSON output:
{
  "scene_type": "border",
  "model_id": "yolo_general",
  "targets": ["person"],
  "rules": [{
    "type": "enter_region",
    "params": {
      "region_id": "restricted_zone",
      "polygon": [[160,120],[480,120],[480,360],[160,360]]
    }
  }],
  "alert": {"enabled": true, "cooldown_seconds": 5}
}
"""


def build_scene_messages(
    user_text: str, *, video_fps: float | None = None,
) -> list[dict[str, str]]:
    """Build the two-message request consumed by DeepSeek JSON mode."""
    system_prompt = SYSTEM_PROMPT
    if video_fps is not None:
        if not math.isfinite(video_fps) or video_fps <= 0:
            raise ValueError("video_fps must be finite and positive")
        system_prompt += (
            f"\nActual uploaded video frame rate: {video_fps:g} FPS. Every frame is processed.\n"
            "For a fire/smoke duration given in seconds, output the seconds parameter; do not calculate frames. "
            "The application will convert to frames using "
            "ceil(seconds * actual FPS) + 1, so the time from the first detection "
            "to confirmation is at least the requested duration. Do not assume 30 FPS. "
            "For a duration already given in frames, keep that exact frame count. "
            "Use max_gap_frames=0 for strictly continuous detection unless the user "
            "explicitly permits gaps. Border dwell rules keep their seconds parameter.\n"
            f"For example, 8 seconds at {video_fps:g} FPS requires {math.ceil(8 * video_fps) + 1} frames.\n"
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

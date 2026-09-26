"""Constrained prompts for converting user requests into SceneSpec JSON."""


SYSTEM_PROMPT = """You convert Chinese video-monitoring requests into one JSON object.
Return JSON only. Do not use Markdown fences, prose, code, commands, or file paths.

Allowed border configuration:
- scene_type: "border"
- model_id: "yolo_general"
- targets: "person", "car", "motorcycle", "bus", "truck"
- rules: "enter_region", "leave_region", "dwell"
- region params: region_id and polygon
- dwell also allows seconds

Allowed fire configuration:
- scene_type: "fire"
- model_id: "fire_smoke"
- targets: "fire", "smoke"
- rule: "consecutive_frames"
- params: rule_id, frames, max_gap_frames

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


def build_scene_messages(user_text: str) -> list[dict[str, str]]:
    """Build the two-message request consumed by DeepSeek JSON mode."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

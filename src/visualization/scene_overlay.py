"""Render configured regions and short-lived event notices on video frames."""

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.rules.base_rule import EventCandidate
from src.schemas.detection import Detection
from src.schemas.scene_spec import SceneSpec
from src.schemas.face_observation import FaceObservation
from src.rules.progress import RuleProgress
from .annotator import Annotator


# OpenCV uses BGR colors.
REGION_COLOR = (0, 200, 255)
ALERT_COLOR = (55, 65, 240)
SUSPECTED_COLOR = (0, 165, 255)
EVENT_LABELS = {
    "post_unstaffed": ("岗位缺员", "Post understaffed"),
    "post_recovered": ("回岗恢复", "Post recovered"),
    "drowsiness_suspected": ("疑似打瞌睡：持续闭眼", "Sustained eye closure"),
    "eyes_reopened": ("睁眼恢复", "Eyes reopened"),
    "enter_region": ("进入禁区", "Entered restricted area"),
    "leave_region": ("离开禁区", "Left restricted area"),
    "dwell": ("区域滞留", "Dwell alert"),
    "fire_suspected": ("疑似火灾", "Fire suspected"),
    "fire_confirmed": ("确认火灾", "Fire confirmed"),
}
TARGET_LABELS = {
    "person": "人员", "car": "车辆", "fire": "火焰", "smoke": "烟雾",
}


@lru_cache(maxsize=12)
def overlay_font(size: int):
    """Use a local Chinese font when available; otherwise show English labels."""
    for filename in (
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(filename).is_file():
            try:
                return ImageFont.truetype(filename, size), True
            except OSError:
                continue
    return ImageFont.load_default(size=size), False


def fit_text(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> str:
    """Keep long region IDs and event labels inside the visible panel."""
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "...", font=font) > width:
        text = text[:-1]
    return text + "..." if text else ""


class SceneOverlay:
    """Track event display lifetimes using video time, independently of rules."""

    def __init__(self, scene: SceneSpec, display_seconds: float = 3.0) -> None:
        if display_seconds <= 0:
            raise ValueError("display_seconds must be positive")
        self.display_seconds = display_seconds
        self._active: list[EventCandidate] = []
        self.progress: list[RuleProgress] = []
        self.face_observation: FaceObservation | None = None
        self.face_observations: tuple[FaceObservation, ...] = ()
        self.scene_type = scene.scene_type
        self.region_names = {r.params.get("region_id",f"{r.type}:{i}"): r.params.get("name") for i,r in enumerate(scene.rules)}
        self.regions: list[tuple[str, np.ndarray]] = []
        seen = set()
        for index, rule in enumerate(scene.rules):
            if rule.type not in {"enter_region", "leave_region", "dwell", "region_understaffed"}:
                continue
            polygon = tuple(tuple(point) for point in rule.params["polygon"])
            region_id = str(rule.params.get("region_id", f"{rule.type}:{index}"))
            identity = (region_id, polygon)
            if identity not in seen:
                seen.add(identity)
                self.regions.append((region_id, np.asarray(polygon, dtype=np.int32)))

    @property
    def active_events(self) -> tuple[EventCandidate, ...]:
        return tuple(self._active)

    def update(self, timestamp_seconds: float, events: Sequence[EventCandidate]) -> None:
        self._active = [
            event for event in self._active
            if 0 <= timestamp_seconds - event.timestamp_seconds < self.display_seconds
        ]
        for event in events:
            if event.event_type in {"post_recovered", "eyes_reopened"}:
                self._active = [old for old in self._active if not (
                    old.trigger_rule == event.trigger_rule and
                    (event.event_type == "post_recovered" or old.track_id == event.track_id))]
            # A confirmation replaces the earlier suspicion for the same episode.
            if event.event_type == "fire_confirmed":
                self._active = [
                    old for old in self._active
                    if not (old.trigger_rule == event.trigger_rule and old.event_type == "fire_suspected")
                ]
            identity = (event.trigger_rule, event.event_type, event.track_id)
            self._active = [
                old for old in self._active
                if (old.trigger_rule, old.event_type, old.track_id) != identity
            ]
            self._active.append(event)

    def draw(self, frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with three channels")
        annotated = frame.copy()
        active_regions = {event.trigger_rule for event in self._active}
        region_labels = []
        for index, (region_id, polygon) in enumerate(self.regions, start=1):
            color = ALERT_COLOR if region_id in active_regions else REGION_COLOR
            if self.scene_type == "on_duty":
                state = next((p.status for p in self.progress if p.rule_id == region_id), "unknown")
                color = {"normal": (70, 190, 70), "tracking": REGION_COLOR,
                         "confirmed": ALERT_COLOR, "unknown": (150,150,150)}.get(state, REGION_COLOR)
            tinted = annotated.copy()
            cv2.fillPoly(tinted, [polygon], color)
            annotated = cv2.addWeighted(tinted, 0.10, annotated, 0.90, 0)
            cv2.polylines(annotated, [polygon], True, color, 2, cv2.LINE_AA)
            region_labels.append((index, region_id, polygon, color))

        annotated = Annotator.draw(annotated, detections)
        faces = self.face_observations or ((self.face_observation,) if self.face_observation else ())
        face_labels = []
        for face in faces:
            progress = next((p for p in self.progress if p.track_id == face.track_id), None)
            state = progress.status if progress else "unknown"
            color = {"normal": (70,190,70), "confirmed": ALERT_COLOR,
                     "unknown": (150,150,150)}.get(state, REGION_COLOR)
            if face.bbox_xyxy:
                x1,y1,x2,y2 = map(int, face.bbox_xyxy)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                face_labels.append((face, progress, color))
            for x,y in face.eye_points:
                cv2.circle(annotated, (round(x),round(y)), 2, color, -1)
        for detection in detections:
            matching = [
                event for event in self._active
                if (event.track_id is not None and event.track_id == detection.track_id)
                or (event.track_id is None and event.target_class == detection.class_name)
            ]
            if matching:
                color = ALERT_COLOR if any(event.alert_status == "confirmed" for event in matching) else SUSPECTED_COLOR
                x1, y1, x2, y2 = map(int, detection.bbox_xyxy)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        if not region_labels and not self._active and not self.progress:
            return annotated
        height, width = annotated.shape[:2]
        font_size = max(10, min(22, round(width / 36)))
        font, chinese = overlay_font(font_size)
        image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        line_height = font_size + 12
        for face, progress, color in face_labels:
            x = max(0, min(int(face.bbox_xyxy[0]), width - 1))
            y = max(0, min(int(face.bbox_xyxy[1]) - line_height, height - line_height))
            status = progress.status if progress else "unknown"
            label = {"normal": "睁眼", "tracking": "闭眼", "confirmed": "报警",
                     "waiting": "待判断", "unknown": "无法判断"}.get(status, status) if chinese else status
            text = f"ID:{face.track_id} {label}"
            if progress and status in {"tracking", "confirmed"}:
                text += f" {progress.current:.1f}/{progress.required:g}s"
            text = fit_text(draw, text, font, max(0, width - x - 12))
            label_width = min(width - x, int(draw.textlength(text, font=font)) + 12)
            draw.rectangle((x, y, x + label_width, y + line_height), fill=(23, 31, 44))
            draw.text((x + 6, y + 3), text, font=font, fill=tuple(reversed(color)))
        for index, region_id, polygon, color in region_labels:
            x = max(0, min(int(polygon[:, 0].min()), width - 1))
            y = max(0, min(int(polygon[:, 1].min()) - line_height, height - line_height))
            text = (self.region_names.get(region_id) or f"禁区 {index} | {region_id}") if chinese else f"Zone {index} | {region_id}"
            text = fit_text(draw, text, font, max(0, width - x - 12))
            label_width = min(width - x, int(draw.textlength(text, font=font)) + 12)
            draw.rectangle((x, y, x + label_width, y + line_height), fill=(23, 31, 44))
            draw.text((x + 6, y + 3), text, font=font, fill=tuple(reversed(color)))

        # Keep notices compact and prioritize confirmed alerts over suspicions.
        ordered = sorted(self._active, key=lambda item: (item.alert_status == "confirmed", item.timestamp_seconds), reverse=True)
        margin = max(2, min(12, width // 40))
        max_rows = max(1, min(3, (height // 3 - 2 * margin) // line_height))
        visible = ordered[:max_rows]
        if visible:
            panel_bottom = margin + len(visible) * line_height + 8
            draw.rounded_rectangle((margin, margin, width - margin, panel_bottom), radius=5, fill=(23, 31, 44))
            for index, event in enumerate(visible):
                color = ALERT_COLOR if event.alert_status == "confirmed" else SUSPECTED_COLOR
                label = EVENT_LABELS.get(event.event_type, (event.event_type, event.event_type))[0 if chinese else 1]
                target = TARGET_LABELS.get(event.target_class, event.target_class) if chinese else event.target_class
                target += f" ID:{event.track_id}" if event.track_id is not None else ""
                text = f"{label}  |  {target}  |  {event.timestamp_seconds:.2f}s"
                if index == len(visible) - 1 and len(ordered) > len(visible):
                    text += f"  (+{len(ordered) - len(visible)})"
                y = margin + 4 + index * line_height
                draw.rectangle((margin + 5, y + 3, margin + 8, y + font_size + 3), fill=tuple(reversed(color)))
                draw.text((margin + 16, y), fit_text(draw, text, font, width - 2 * margin - 24), font=font, fill=tuple(reversed(color)))
        if self.progress and height >= 120 and width >= 160:
            statuses = sorted(self.progress,key=lambda p:(p.status in {"tracking","confirmed","cooldown"},p.current/max(p.required,0.001)),reverse=True)
            shown = statuses[:min(3,max(1,height//(line_height*5)))]
            bottom = height-margin
            top = bottom-len(shown)*(line_height+4)-8
            draw.rounded_rectangle((margin,top,width-margin,bottom),radius=5,fill=(23,31,44))
            for i,item in enumerate(shown):
                y=top+3+i*(line_height+4)
                text=item.text if chinese else f"{item.rule_id} ID:{item.track_id} {item.current:.1f}/{item.required:g} {item.status}"
                if i==len(shown)-1 and len(statuses)>len(shown):
                    text=f"(+{len(statuses)-len(shown)}) "+text
                draw.text((margin+8,y),fit_text(draw,text,font,width-2*margin-16),font=font,fill=(220,238,246))
                ratio=min(1,max(0,item.current/max(item.required,0.001)))
                x1=margin+8
                x2=width-margin-8
                draw.line((x1,y+line_height,x2,y+line_height),fill=(65,80,98),width=3)
                if ratio:
                    draw.line((x1,y+line_height,x1+(x2-x1)*ratio,y+line_height),fill=(255,121,99) if item.status=="confirmed" else (70,216,187),width=3)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

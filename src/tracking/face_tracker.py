"""Conservative adjacent-frame face association; IDs are not person identities."""

from src.schemas.face_observation import FaceObservation


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1, area_a + area_b - intersection)


class FaceTracker:
    def __init__(self, min_iou=0.2, max_gap_seconds=0.5):
        self.min_iou = min_iou
        self.max_gap_seconds = max_gap_seconds
        self._previous = {}
        self._last_time = None
        self._next_id = 1

    def update(self, faces: list[FaceObservation], timestamp_seconds: float):
        if self._last_time is not None and (
            timestamp_seconds <= self._last_time or timestamp_seconds - self._last_time > self.max_gap_seconds
        ):
            self._previous.clear()
        self._last_time = timestamp_seconds
        # Only accept mutually unique plausible matches. Crossings/overlap start
        # new tracks instead of transferring accumulated closure to another face.
        candidates = {
            i: [track_id for track_id, box in self._previous.items()
                if face.bbox_xyxy and box_iou(face.bbox_xyxy, box) >= self.min_iou]
            for i, face in enumerate(faces)
        }
        uses = {}
        for ids in candidates.values():
            for track_id in ids:
                uses[track_id] = uses.get(track_id, 0) + 1
        tracked = []
        for i, face in enumerate(faces):
            ids = candidates[i]
            continuous = len(ids) == 1 and uses[ids[0]] == 1
            if continuous:
                track_id = ids[0]
            else:
                track_id = self._next_id
                self._next_id += 1
            tracked.append(face.model_copy(update={"track_id": track_id, "continuous": continuous}))
        # No bridging missing frames: disappearance immediately breaks continuity.
        self._previous = {f.track_id: f.bbox_xyxy for f in tracked if f.bbox_xyxy}
        return tuple(tracked)

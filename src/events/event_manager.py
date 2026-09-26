"""Convert rule candidates into formal events and evidence snapshots."""

from pathlib import Path

import cv2
import numpy as np

from src.rules.base_rule import EventCandidate
from src.schemas.event import Event
from src.schemas.scene_spec import SceneType


class EventManager:
    """Assign event IDs, reject exact duplicates, and save confirmed evidence."""

    def __init__(
        self,
        run_dir: str | Path,
        scene_type: SceneType,
        model_id: str,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.scene_type = scene_type
        self.model_id = model_id
        self._next_id = 1
        self._seen: set[tuple[int, str, int | None, str]] = set()

    def add(self, candidate: EventCandidate, frame: np.ndarray) -> Event | None:
        """Build one event, or return ``None`` for an exact duplicate candidate."""
        key = (
            candidate.frame_id,
            candidate.event_type,
            candidate.track_id,
            candidate.trigger_rule,
        )
        if key in self._seen:
            return None

        event_id = f"evt-{self._next_id:06d}"
        snapshot_path = None
        if candidate.alert_status == "confirmed":
            snapshot_path = self._save_snapshot(event_id, frame)

        event = Event(
            event_id=event_id,
            scene_type=self.scene_type,
            event_type=candidate.event_type,
            target_class=candidate.target_class,
            track_id=candidate.track_id,
            timestamp_seconds=candidate.timestamp_seconds,
            confidence=candidate.confidence,
            trigger_rule=candidate.trigger_rule,
            snapshot_path=snapshot_path,
            alert_status=candidate.alert_status,
            model_id=self.model_id,
        )
        self._seen.add(key)
        self._next_id += 1
        return event

    def _save_snapshot(self, event_id: str, frame: np.ndarray) -> str:
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
        ):
            raise ValueError("confirmed event frame must be a non-empty BGR image")

        encoded_ok, encoded = cv2.imencode(".jpg", frame)
        if not encoded_ok or encoded is None:
            raise OSError("failed to encode event snapshot")

        relative_path = Path("snapshots") / f"{event_id}.jpg"
        snapshot_path = self.run_dir / relative_path
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            snapshot_path.write_bytes(encoded.tobytes())
        except OSError as error:
            snapshot_path.unlink(missing_ok=True)
            raise OSError(f"failed to write event snapshot: {snapshot_path}") from error

        return relative_path.as_posix()

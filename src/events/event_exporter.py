"""Export formal events and run metadata as reproducible evidence files."""

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from src.schemas.event import Event


CSV_FIELDS = [
    "event_id",
    "scene_type",
    "event_type",
    "target_class",
    "track_id",
    "timestamp_seconds",
    "confidence",
    "trigger_rule",
    "snapshot_path",
    "alert_status",
    "model_id",
]


class EventExporter:
    """Write config, event rows, and summary files into one run directory."""

    def export(
        self,
        run_dir: str | Path,
        config: Any,
        events: Sequence[Event],
        metrics: Mapping[str, Any],
    ) -> None:
        """Serialize all outputs first, then atomically replace each target file."""
        payloads = self._build_payloads(config, events, metrics)
        output_dir = Path(run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        temporary_paths: dict[str, Path] = {}
        try:
            for filename, payload in payloads.items():
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=output_dir,
                    prefix=f".{filename}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_paths[filename] = Path(temporary_file.name)
                    temporary_file.write(payload)
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())

            for filename, temporary_path in temporary_paths.items():
                os.replace(temporary_path, output_dir / filename)
        finally:
            for temporary_path in temporary_paths.values():
                temporary_path.unlink(missing_ok=True)

    def _build_payloads(
        self,
        config: Any,
        events: Sequence[Event],
        metrics: Mapping[str, Any],
    ) -> dict[str, bytes]:
        event_rows = [event.model_dump(mode="json") for event in events]
        config_data = self._config_data(config)
        status_counts = Counter(event.alert_status for event in events)
        event_type_counts = Counter(event.event_type for event in events)
        summary = {
            "event_count": len(events),
            "confirmed_count": status_counts["confirmed"],
            "suspected_count": status_counts["suspected"],
            "event_type_counts": dict(event_type_counts),
            "metrics": dict(metrics),
        }

        return {
            "config.json": self._json_bytes(config_data),
            "events.json": self._json_bytes(event_rows),
            "events.csv": self._csv_bytes(events),
            "summary.json": self._json_bytes(summary),
        }

    @staticmethod
    def _config_data(config: Any) -> dict[str, Any]:
        model_dump = getattr(config, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json")
        if isinstance(config, Mapping):
            return dict(config)
        raise TypeError("config must be a Pydantic model or mapping")

    @staticmethod
    def _json_bytes(value: Any) -> bytes:
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        return text.encode("utf-8")

    @staticmethod
    def _csv_bytes(events: Sequence[Event]) -> bytes:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for event in events:
            row = event.model_dump(mode="json")
            row["track_id"] = "" if event.track_id is None else event.track_id
            row["timestamp_seconds"] = f"{event.timestamp_seconds:.3f}"
            row["snapshot_path"] = event.snapshot_path or ""
            writer.writerow(row)
        return buffer.getvalue().encode("utf-8-sig")

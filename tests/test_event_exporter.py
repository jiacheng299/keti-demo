import csv
import json

import pytest

import src.events.event_exporter as event_exporter_module
from src.events.event_exporter import EventExporter
from src.schemas.event import Event
from src.schemas.scene_spec import SceneSpec


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


def make_scene_spec() -> SceneSpec:
    return SceneSpec.model_validate(
        {
            "scene_type": "fire",
            "model_id": "fire_smoke",
            "targets": ["fire", "smoke"],
            "rules": [
                {
                    "type": "consecutive_frames",
                    "params": {"frames": 8, "max_gap_frames": 1},
                }
            ],
        }
    )


def make_event(
    event_id: str,
    timestamp: float,
    *,
    event_type: str = "fire_confirmed",
    alert_status: str = "confirmed",
) -> Event:
    return Event(
        event_id=event_id,
        scene_type="fire",
        event_type=event_type,
        target_class="fire",
        track_id=None,
        timestamp_seconds=timestamp,
        confidence=0.91,
        trigger_rule="fire_sequence",
        snapshot_path=(
            f"snapshots/{event_id}.jpg" if alert_status == "confirmed" else None
        ),
        alert_status=alert_status,
        model_id="fire_smoke",
    )


def test_exporter_writes_matching_json_csv_config_and_summary(tmp_path):
    events = [
        make_event("evt-000001", 1.23456),
        make_event(
            "evt-000002",
            2.0,
            event_type="fire_suspected",
            alert_status="suspected",
        ),
    ]

    EventExporter().export(
        tmp_path,
        config=make_scene_spec(),
        events=events,
        metrics={"processed_frames": 100, "elapsed_seconds": 8.42},
    )

    config = json.loads((tmp_path / "config.json").read_text("utf-8"))
    json_rows = json.loads((tmp_path / "events.json").read_text("utf-8"))
    with (tmp_path / "events.csv").open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        csv_rows = list(reader)
        assert reader.fieldnames == CSV_FIELDS
    summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))

    assert config["scene_type"] == "fire"
    assert [row["event_id"] for row in json_rows] == ["evt-000001", "evt-000002"]
    assert [row["event_id"] for row in csv_rows] == ["evt-000001", "evt-000002"]
    assert csv_rows[0]["timestamp_seconds"] == "1.235"
    assert csv_rows[0]["track_id"] == ""
    assert csv_rows[1]["snapshot_path"] == ""
    assert (tmp_path / "events.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert summary == {
        "event_count": 2,
        "confirmed_count": 1,
        "suspected_count": 1,
        "event_type_counts": {"fire_confirmed": 1, "fire_suspected": 1},
        "metrics": {"processed_frames": 100, "elapsed_seconds": 8.42},
    }


def test_exporter_writes_valid_empty_outputs(tmp_path):
    EventExporter().export(tmp_path, make_scene_spec(), [], {})

    assert json.loads((tmp_path / "events.json").read_text("utf-8")) == []
    with (tmp_path / "events.csv").open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == CSV_FIELDS
        assert list(reader) == []
    summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))
    assert summary["event_count"] == 0


def test_exporter_does_not_cross_write_run_directories(tmp_path):
    first = tmp_path / "run-one"
    second = tmp_path / "run-two"
    exporter = EventExporter()

    exporter.export(first, make_scene_spec(), [make_event("evt-000001", 1.0)], {})
    exporter.export(second, make_scene_spec(), [], {})

    assert len(json.loads((first / "events.json").read_text("utf-8"))) == 1
    assert json.loads((second / "events.json").read_text("utf-8")) == []


def test_exporter_preserves_existing_files_when_serialization_fails(tmp_path):
    exporter = EventExporter()
    exporter.export(tmp_path, make_scene_spec(), [make_event("evt-000001", 1.0)], {})
    before = {
        name: (tmp_path / name).read_bytes()
        for name in ("config.json", "events.json", "events.csv", "summary.json")
    }

    with pytest.raises(TypeError):
        exporter.export(
            tmp_path,
            make_scene_spec(),
            [make_event("evt-000002", 2.0)],
            {"not_json_serializable": object()},
        )

    after = {name: (tmp_path / name).read_bytes() for name in before}
    assert after == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_exporter_accepts_plain_mapping_config(tmp_path):
    config = {
        "scene_type": "fire",
        "model_id": "fire_smoke",
        "targets": ["fire"],
        "rules": [],
    }

    EventExporter().export(tmp_path, config, [], {})

    exported = json.loads((tmp_path / "config.json").read_text("utf-8"))
    assert exported == config


def test_exporter_cleans_temporary_files_when_temp_write_fails(tmp_path, monkeypatch):
    exporter = EventExporter()
    exporter.export(tmp_path, make_scene_spec(), [make_event("evt-000001", 1.0)], {})
    before = {
        name: (tmp_path / name).read_bytes()
        for name in ("config.json", "events.json", "events.csv", "summary.json")
    }
    real_fsync = event_exporter_module.os.fsync
    call_count = 0

    def fail_second_fsync(file_descriptor):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("simulated temporary write failure")
        real_fsync(file_descriptor)

    monkeypatch.setattr(event_exporter_module.os, "fsync", fail_second_fsync)

    with pytest.raises(OSError, match="simulated temporary write failure"):
        exporter.export(tmp_path, make_scene_spec(), [make_event("evt-000002", 2.0)], {})

    after = {name: (tmp_path / name).read_bytes() for name in before}
    assert after == before
    assert list(tmp_path.glob("*.tmp")) == []

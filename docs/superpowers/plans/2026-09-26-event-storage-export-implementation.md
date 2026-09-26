# Event Storage and Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `EventCandidate` objects into deduplicated formal events with confirmed-event screenshots, then export reproducible JSON, CSV, configuration, and summary files.

**Architecture:** `EventManager` owns event identity, exact duplicate protection, and confirmed-event screenshots. `EventExporter` owns deterministic serialization and per-file atomic replacement of the four export files. Rule cooldown remains in `RuleEngine`; this feature does not add a second time-window filter.

**Tech Stack:** Python 3.11, Pydantic, NumPy, OpenCV, standard-library `json`, `csv`, `tempfile`, `os`, and pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-event-storage-export-design.md`

## Global Constraints

- Run on the existing CPU-only Windows environment.
- Save screenshots only for events whose `alert_status` is `confirmed`.
- Keep suspected events with `snapshot_path=None`.
- Store snapshot paths relative to the run directory using `/` separators.
- Keep time-window cooldown exclusively in `RuleEngine`.
- Use `utf-8` JSON and `utf-8-sig` CSV.
- Format CSV `timestamp_seconds` with exactly three decimal places.
- Do not add database, result-video, Streamlit, ZIP, or cleanup features.
- Do not commit implementation until the user completes code learning and acceptance.

---

### Task 1: Formal Event Creation and Screenshots

**Files:**
- Create: `src/events/__init__.py`
- Create: `src/events/event_manager.py`
- Create: `tests/test_event_manager.py`

**Interfaces:**
- Consumes: `EventCandidate`, `Event`, `SceneType`, and a BGR `numpy.ndarray` frame.
- Produces: `EventManager(run_dir: str | Path, scene_type: SceneType, model_id: str)` and `EventManager.add(candidate: EventCandidate, frame: np.ndarray) -> Event | None`.

- [x] **Step 1: Write failing tests for conversion, numbering, and exact deduplication**

```python
def make_candidate(
    *,
    frame_id: int = 7,
    event_type: str = "fire_suspected",
    alert_status: str = "suspected",
    track_id: int | None = None,
) -> EventCandidate:
    return EventCandidate(
        frame_id=frame_id,
        timestamp_seconds=frame_id / 25,
        event_type=event_type,
        target_class="fire" if track_id is None else "person",
        track_id=track_id,
        confidence=0.91,
        trigger_rule="fire_sequence" if track_id is None else "restricted_zone",
        alert_status=alert_status,
    )


def test_event_manager_builds_events_and_deduplicates(tmp_path):
    manager = EventManager(tmp_path, "fire", "fire_smoke")
    candidate = EventCandidate(
        frame_id=7,
        timestamp_seconds=0.28,
        event_type="fire_suspected",
        target_class="fire",
        track_id=None,
        confidence=0.91,
        trigger_rule="fire_sequence",
        alert_status="suspected",
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    event = manager.add(candidate, frame)
    duplicate = manager.add(candidate, frame)

    assert event.event_id == "evt-000001"
    assert event.scene_type == "fire"
    assert event.model_id == "fire_smoke"
    assert event.snapshot_path is None
    assert duplicate is None
```

- [x] **Step 2: Write failing tests for confirmed screenshots and Track ID preservation**

```python
def test_confirmed_event_saves_readable_snapshot(tmp_path):
    manager = EventManager(tmp_path, "border", "yolo_general")
    candidate = make_candidate(
        frame_id=12,
        event_type="enter_region",
        alert_status="confirmed",
        track_id=4,
    )
    frame = np.full((32, 32, 3), 127, dtype=np.uint8)

    event = manager.add(candidate, frame)

    assert event.track_id == 4
    assert event.snapshot_path == "snapshots/evt-000001.jpg"
    assert cv2.imread(str(tmp_path / event.snapshot_path)) is not None
```

- [x] **Step 3: Write a failing test for invalid screenshot frames**

```python
def test_confirmed_event_rejects_invalid_frame_without_consuming_id(tmp_path):
    manager = EventManager(tmp_path, "fire", "fire_smoke")
    candidate = make_candidate(alert_status="confirmed")

    with pytest.raises(ValueError, match="BGR"):
        manager.add(candidate, np.zeros((32, 32), dtype=np.uint8))

    event = manager.add(candidate, np.zeros((32, 32, 3), dtype=np.uint8))
    assert event.event_id == "evt-000001"
```

- [x] **Step 4: Run tests and verify missing-module failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_event_manager.py -q
```

Expected: collection fails because `src.events.event_manager` does not exist.

- [x] **Step 5: Implement `EventManager` minimally**

Implementation requirements:

```python
class EventManager:
    def __init__(self, run_dir, scene_type, model_id):
        self.run_dir = Path(run_dir)
        self.scene_type = scene_type
        self.model_id = model_id
        self._next_id = 1
        self._seen: set[tuple[int, str, int | None, str]] = set()

    def add(self, candidate, frame):
        key = (
            candidate.frame_id,
            candidate.event_type,
            candidate.track_id,
            candidate.trigger_rule,
        )
        if key in self._seen:
            return None
        event_id = f"evt-{self._next_id:06d}"
        snapshot_path = self._save_snapshot(event_id, frame) \
            if candidate.alert_status == "confirmed" else None
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
```

Use `cv2.imencode(".jpg", frame)` plus `Path.write_bytes` so Chinese Windows paths remain writable; validate `frame.ndim == 3` and `frame.shape[2] == 3`; create `snapshots/` only when saving the first confirmed event; raise `OSError` when encoding or writing fails.

- [x] **Step 6: Run Task 1 tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_event_manager.py -q
```

Expected: all Task 1 tests pass.

---

### Task 2: Deterministic Exports and Per-file Atomic Replacement

**Files:**
- Create: `src/events/event_exporter.py`
- Modify: `src/events/__init__.py`
- Create: `tests/test_event_exporter.py`

**Interfaces:**
- Consumes: `SceneSpec | Mapping[str, Any]`, `Sequence[Event]`, and `Mapping[str, Any]` metrics.
- Produces: `EventExporter.export(run_dir, config, events, metrics) -> None` and the four files `config.json`, `events.json`, `events.csv`, and `summary.json`.

- [x] **Step 1: Write a failing test for all export files and stable CSV formatting**

```python
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


def make_event(event_id: str, timestamp: float) -> Event:
    return Event(
        event_id=event_id,
        scene_type="fire",
        event_type="fire_confirmed",
        target_class="fire",
        track_id=None,
        timestamp_seconds=timestamp,
        confidence=0.91,
        trigger_rule="fire_sequence",
        snapshot_path=f"snapshots/{event_id}.jpg",
        alert_status="confirmed",
        model_id="fire_smoke",
    )


def test_exporter_writes_matching_json_csv_and_summary(tmp_path):
    exporter = EventExporter()
    events = [make_event("evt-000001", 1.23456)]

    exporter.export(
        tmp_path,
        config=make_scene_spec(),
        events=events,
        metrics={"processed_frames": 100, "elapsed_seconds": 8.42},
    )

    json_rows = json.loads((tmp_path / "events.json").read_text("utf-8"))
    with (tmp_path / "events.csv").open(encoding="utf-8-sig", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))

    assert len(json_rows) == len(csv_rows) == 1
    assert csv_rows[0]["timestamp_seconds"] == "1.235"
    assert summary["event_count"] == 1
    assert summary["confirmed_count"] == 1
```

- [x] **Step 2: Write a failing test for empty events**

```python
def test_exporter_writes_valid_empty_outputs(tmp_path):
    EventExporter().export(tmp_path, make_scene_spec(), [], {})

    assert json.loads((tmp_path / "events.json").read_text("utf-8")) == []
    with (tmp_path / "events.csv").open(encoding="utf-8-sig", newline="") as file:
        assert list(csv.DictReader(file)) == []
    summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))
    assert summary["event_count"] == 0
```

- [x] **Step 3: Write a failing test for separate run directories**

```python
def test_exporter_does_not_cross_write_run_directories(tmp_path):
    first = tmp_path / "run-one"
    second = tmp_path / "run-two"
    exporter = EventExporter()

    exporter.export(first, make_scene_spec(), [make_event("evt-000001", 1.0)], {})
    exporter.export(second, make_scene_spec(), [], {})

    assert len(json.loads((first / "events.json").read_text("utf-8"))) == 1
    assert json.loads((second / "events.json").read_text("utf-8")) == []
```

- [x] **Step 4: Run tests and verify missing-module failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_event_exporter.py -q
```

Expected: collection fails because `src.events.event_exporter` does not exist.

- [x] **Step 5: Implement deterministic serialization**

Use the exact CSV field order from the spec. Convert config with `model_dump(mode="json")` when available, otherwise `dict(config)`. Build all four byte/text payloads in memory before opening output files. Compute summary counts using `Counter` while preserving input event order in JSON and CSV.

- [x] **Step 6: Implement atomic replacement**

For each target, create a temporary file in `run_dir`, write and flush it, then replace the final file with `os.replace`. If serialization or any temporary write fails, remove every temporary file and leave existing final files untouched. Replace final files only after all temporary writes succeed. Each replacement is atomic for one file; the four-file set does not claim transaction-level atomicity.

- [x] **Step 7: Run Task 2 tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_event_exporter.py -q
```

Expected: all Task 2 tests pass.

---

### Task 3: Documentation and Final Verification

**Files:**
- Modify: `docs/03_feature_manual.md`
- Modify: `docs/superpowers/plans/2026-09-17-open-scene-demo-implementation.md`
- Modify: `worklog/daily_checklist.md`
- Modify: `worklog/daily_log.md`

**Interfaces:**
- Consumes: verified Task 1 and Task 2 behavior.
- Produces: accurate feature status, validation evidence, and learning checklist.

- [x] **Step 1: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_event_manager.py tests\test_event_exporter.py -q
```

Expected: all event tests pass.

- [x] **Step 2: Run the complete test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: zero failures.

- [x] **Step 3: Inspect generated artifacts**

Run the focused exporter test with a retained temporary directory or a one-off temporary validation script. Confirm:

```text
config.json
events.json
events.csv
summary.json
snapshots/evt-000001.jpg
```

Open the JPEG with OpenCV and parse all JSON/CSV files before recording evidence.

- [x] **Step 4: Update status and worklog documents**

Record exact test counts, artifact paths, screenshot readability, implemented boundaries, user actual time as pending, and the next task as DeepSeek parser/offline templates.

- [ ] **Step 5: Stop for user learning and acceptance**

Teach in this order:

1. `event_manager.py`
2. `event_exporter.py`
3. JSON/CSV/summary output relationship

Do not create the implementation commit until the user finishes learning and explicitly requests it.

- [ ] **Step 6: Commit after user approval**

```powershell
git add -A
git diff --cached --check
git commit -m "feat(events): add evidence exports"
```

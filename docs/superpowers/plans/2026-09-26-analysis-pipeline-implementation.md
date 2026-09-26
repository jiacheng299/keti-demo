# Analysis Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect video input, local adapters, tracking, rules, event evidence, annotation, result-video writing, and exports through one CPU-safe orchestration API.

**Architecture:** `AnalysisPipeline` creates fresh stateful components per run and delegates all detection, tracking, rule, evidence, drawing, and serialization behavior to existing modules. It writes every source frame to preserve video duration, but only runs inference on frames selected by `FrameSampler`. `RunSummary` and `PipelineProgress` provide stable data for the later Streamlit UI.

**Tech Stack:** Python 3.11, OpenCV, Pydantic contracts, dataclasses, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-open-scene-vision-demo-design.md`

## Global Constraints

- Run on CPU and process only local video files.
- Do not put model-specific inference or rule logic inside the pipeline.
- Create a new `Tracker`, `RuleEngine`, and `EventManager` for every run.
- Track detections only for the `border` scene; fire remains a frame-level event.
- Write every input frame to `result.mp4`; skipped inference frames remain unannotated.
- Save confirmed-event screenshots from the annotated processed frame.
- Export partial results after a normal stop request.
- Always close video resources and unload the model, including error paths.
- Progress callbacks run after each frame is written.
- Do not add UI, background threads, queues, databases, or concurrent videos.
- Do not commit until the user requests the next checkpoint.

---

### Task 1: Pipeline Contracts and Fire End-to-End Flow

**Files:**
- Create: `src/pipeline/__init__.py`
- Create: `src/pipeline/contracts.py`
- Create: `src/pipeline/analysis_pipeline.py`
- Modify: `src/models/base_adapter.py`
- Create: `tests/test_pipeline_smoke.py`

**Interfaces:**
- `ModelAdapter.predict(frame, frame_id: int = 0) -> list[Detection]`.
- `PipelineProgress(frames_read, total_frames, processed_frames, event_count)` with `fraction`.
- `RunSummary(run_dir, result_video_path, events, frames_read, processed_frames, detection_count, stopped, elapsed_seconds)`.
- `AnalysisPipeline.run(video_path, scene_spec, run_dir, progress_callback=None, stop_event=None) -> RunSummary`.

- [x] **Step 1: Write a failing fire-scene pipeline test**

Create a three-frame MJPG AVI and a deterministic adapter that returns one fire detection per call. Use a fire `SceneSpec` requiring two consecutive frames and cooldown zero. Assert:

```python
summary = pipeline.run(video_path, scene, run_dir, progress.append)

assert adapter.loaded is True
assert adapter.unloaded is True
assert [event.alert_status for event in summary.events] == ["suspected", "confirmed"]
assert summary.frames_read == 3
assert summary.processed_frames == 3
assert summary.detection_count == 3
assert summary.stopped is False
assert progress[-1].fraction == 1.0
assert [item for item in adapter.frame_ids] == [0, 1, 2]
```

Parse `events.json`, `events.csv`, and `summary.json`; assert matching event counts. Decode `snapshots/evt-000002.jpg`. Open `result.mp4` and assert it contains three frames.

- [x] **Step 2: Run the fire pipeline test and verify the missing-module failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline_smoke.py::test_fire_pipeline_writes_video_events_and_exports -q
```

- [x] **Step 3: Add contracts and update the adapter protocol**

Use frozen dataclasses. `PipelineProgress.fraction` returns `frames_read / total_frames`, clamped to `[0, 1]`, or zero when total frames is zero. `RunSummary.events` is `tuple[Event, ...]`.

Update the shared protocol to match both existing adapters:

```python
def predict(self, frame: Any, frame_id: int = 0) -> list[Detection]: ...
```

- [x] **Step 4: Implement the minimal orchestration path**

For each processed frame:

```python
detections = adapter.predict(frame, frame_id)
if scene_spec.scene_type == "border":
    detections = tracker.update(detections, timestamp_seconds)
candidates = rule_engine.evaluate(
    scene_spec,
    detections,
    FrameState(frame_id=frame_id, timestamp_seconds=timestamp_seconds),
)
annotated = annotator.draw(frame, detections)
for candidate in candidates:
    event = event_manager.add(candidate, annotated)
    if event is not None:
        events.append(event)
```

Write the frame, emit progress, unload in `finally`, then export metrics and return `RunSummary`.

- [x] **Step 5: Run the fire test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline_smoke.py::test_fire_pipeline_writes_video_events_and_exports -q
```

---

### Task 2: Border Tracking, Sampling, Stop, and Cleanup

**Files:**
- Modify: `tests/test_pipeline_smoke.py`
- Modify: `src/pipeline/analysis_pipeline.py`

- [x] **Step 1: Write a failing border tracking test**

The fake adapter returns a person outside the region on frame zero and inside it on frame one. Assert one `enter_region` event with a non-null Track ID and a readable confirmed screenshot.

- [x] **Step 2: Write a failing sampling test**

Run three input frames with `FrameSampler(every_n_frames=2)`. Assert adapter frame IDs are `[0, 2]`, `processed_frames == 2`, but the result video still contains three frames.

- [x] **Step 3: Write a failing stop-and-cleanup test**

The progress callback sets a `threading.Event` after the first frame. Assert:

```python
assert summary.stopped is True
assert summary.frames_read == 1
assert adapter.unloaded is True
assert json.loads((run_dir / "summary.json").read_text("utf-8"))["metrics"]["stopped"] is True
```

- [x] **Step 4: Implement only behavior required by the new tests**

Check `stop_event.is_set()` before each frame. On a normal stop, exit the loop, close the writer/source, unload the adapter, and export collected events. Keep exception behavior unchanged: close/unload and re-raise.

- [x] **Step 5: Run all pipeline tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline_smoke.py -q
```

---

### Task 3: Default Wiring and Verification

**Files:**
- Modify: `src/pipeline/analysis_pipeline.py`
- Modify: `tests/test_pipeline_smoke.py`
- Modify: `docs/03_feature_manual.md`
- Modify: `docs/superpowers/plans/2026-09-17-open-scene-demo-implementation.md`
- Modify: `worklog/daily_checklist.md`
- Modify: `worklog/daily_log.md`

- [x] **Step 1: Write a failing default-wiring test**

Call `AnalysisPipeline.from_model_config("config/models.yaml")` and assert the returned object has a registry capable of constructing both adapter types after factories are registered internally. Do not call `load()` in this unit test.

- [x] **Step 2: Implement default wiring**

Load `ModelRegistry`, register `YoloAdapter` under `yolo` and `FireAdapter` under `fire`, and return `AnalysisPipeline(registry, frame_sampler=...)`.

- [x] **Step 3: Run focused and complete tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline_smoke.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

- [x] **Step 4: Run a short real-model validation**

Create temporary clips from the first few frames of `assets/demo_videos/vtest.avi` and `assets/demo_videos/fire_burning.ogv`. Run the default pipeline with the matching offline templates and CPU frame sampling. Parse all exports and confirm each result video is readable. Record observed event counts without claiming model accuracy from these short clips.

- [x] **Step 5: Update status and work logs**

Record exact synthetic-test counts, full-suite count, real short-clip paths and event counts, output readability, actual time as pending, and Streamlit integration as the next task.

- [ ] **Step 6: Stop at the Git checkpoint**

Recommend:

```text
feat(pipeline): connect video analysis flow
```

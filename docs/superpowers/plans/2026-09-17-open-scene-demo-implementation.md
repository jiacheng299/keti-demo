# Open Scene Vision Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-friendly, two-scene video analysis demo that turns natural-language requests into validated scene configuration, selects a local model plugin, evaluates events, and exports evidence.

**Architecture:** Keep a shared video pipeline and normalized detection/event contracts. Add border and fire capabilities as independent model adapters and rule configurations. Use DeepSeek only to produce validated `SceneSpec` JSON; provide offline templates when the API is unavailable.

**Tech Stack:** Python 3.11, Streamlit, OpenCV, Ultralytics YOLO, ByteTrack, Pydantic, PyYAML, Pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-open-scene-vision-demo-design.md`

## Global Constraints

- Development period: 14 days, 3 hours per day, about 42 hours total.
- Hardware: CPU-only personal laptop.
- Video input: local files only in this demo.
- LLM: DeepSeek API parses requests; no local multimodal LLM deployment.
- Vision: local lightweight models; public pretrained weights and public demo videos are allowed.
- Do not modify the original project proposal.
- Demo validates core flow; do not claim production completion or formal proposal metrics without evidence.
- Do not execute arbitrary code or commands returned by DeepSeek.
- Keep API keys in environment variables; never commit them.
- Before each production behavior change, add a test and observe the expected failure.
- Freeze features after Day 12; Days 13 and 14 are for verification, fallback preparation, and detailed user documentation.

---

## File Structure

- `app.py`: Streamlit entry point and page composition.
- `src/schemas/scene_spec.py`: validated request and rule data contracts.
- `src/schemas/detection.py`: normalized model output.
- `src/schemas/event.py`: normalized event output.
- `src/llm/deepseek_client.py`: API transport, timeout, and retry behavior.
- `src/llm/scene_parser.py`: prompt response parsing and schema validation.
- `src/llm/prompts.py`: constrained prompt templates.
- `src/models/base_adapter.py`: model adapter protocol.
- `src/models/model_registry.py`: model configuration lookup and adapter selection.
- `src/models/yolo_adapter.py`: generic YOLO adapter for people and vehicles.
- `src/models/fire_adapter.py`: fire and smoke adapter.
- `src/video/video_source.py`: local video validation and metadata.
- `src/video/frame_sampler.py`: CPU-aware frame sampling.
- `src/video/video_writer.py`: annotated output video writing.
- `src/tracking/tracker.py`: short-lived track IDs and center-point history.
- `src/rules/region_rules.py`: region entry and exit rules.
- `src/rules/temporal_rules.py`: dwell and consecutive-frame rules.
- `src/rules/rule_engine.py`: evaluate configured rules against normalized evidence.
- `src/events/event_manager.py`: deduplication, cooldown, and snapshots.
- `src/events/event_exporter.py`: JSON, CSV, and run summary exports.
- `src/visualization/annotator.py`: draw detections, zones, and alerts.
- `src/pipeline/analysis_pipeline.py`: orchestrate the validated flow.
- `config/settings.yaml`: application defaults.
- `config/models.yaml`: model IDs, local weight paths, classes, and CPU settings.
- `config/scenes/*.yaml`: offline scene templates.
- `tests/`: unit and integration tests.
- `worklog/daily_checklist.md`: fourteen-day progress checklist.
- `worklog/daily_log.md`: actual time, evidence, issues, and next action.
- `docs/03_feature_manual.md`: user-facing function description.
- `docs/05_test_and_acceptance.md`: test cases and defense acceptance criteria.
- `docs/04_user_guide.md`: installation, controls, complete workflows, outputs, troubleshooting, and offline fallback.

## Day 1: Project Foundation

### Task 1: Create a reproducible Python project shell

**Files:**
- Create: `app.py`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `README.md`
- Create: `src/__init__.py`
- Create: `config/settings.yaml`
- Test: `tests/test_app_smoke.py`

**Interfaces:**
- `app.py` exposes `main() -> None` and runs it under `if __name__ == "__main__"`.
- `main()` renders the title `开放场景视觉语义认知 Demo` and a short health/status section.
- `.env.example` contains only `DEEPSEEK_API_KEY=` with no real credential.

- [x] **Step 1: Write the failing smoke test**

Create `tests/test_app_smoke.py`:

```python
from streamlit.testing.v1 import AppTest


def test_app_renders_title_and_mode_status():
    app = AppTest.from_file("app.py").run()

    assert not app.exception
    assert any("开放场景视觉语义认知 Demo" in item.value for item in app.title)
    assert any("当前阶段：项目骨架" in item.value for item in app.markdown)
```

- [x] **Step 2: Run the smoke test and confirm expected failure**

Run: `python -m pytest tests/test_app_smoke.py -q`
Expected: FAIL because `app.py` does not exist.

- [x] **Step 3: Add the minimal app and configuration**

Create `app.py`:

```python
import streamlit as st


def main() -> None:
    st.title("开放场景视觉语义认知 Demo")
    st.markdown("当前阶段：项目骨架。")


if __name__ == "__main__":
    main()
```

Add only required dependencies to `requirements.txt`: `streamlit`, `pytest`. Add a minimal YAML settings file with app title, upload size limit, and default CPU frame size. Document environment creation and app launch in README.

- [x] **Step 4: Run the smoke test and verify it passes**

Run: `python -m pytest tests/test_app_smoke.py -q`
Expected: PASS, one test.

- [x] **Step 5: Launch the app locally**

Run: `streamlit run app.py --server.headless true`
Expected: Streamlit starts without import errors and serves the page locally. Stop it with Ctrl+C after confirming.

### Task 2: Initialize the daily work log

**Files:**
- Modify: `worklog/daily_checklist.md`
- Modify: `worklog/daily_log.md`

**Interfaces:**
- Checklist has one date, three-hour budget, tasks, evidence, and completion status per day.
- Daily log captures actual hours, AI-assisted changes, verification evidence, blockers, and next action.

- [x] **Step 1: Mark Day 1 work items and evidence fields**
- [x] **Step 2: Leave personal time unfilled until the user reports it**
- [x] **Step 3: Add time categories for daily totals**

## Day 2: Border Model Baseline

### Task 3: Verify generic people and vehicle detection

**Files:** `models/border/` (ignored weights), `scripts/validate_border_model.py`, `tests/test_model_output.py`, `worklog/daily_log.md`.

**Interfaces:** validation script accepts `--video`, `--weights`, and `--imgsz`; prints readable CPU timing and writes an annotated sample.

- [x] Test normalized bounding-box conversion on a fixed in-memory prediction fixture.
- [x] Run test and observe failure before implementation.
- [x] Download a public lightweight general detector and record source, license, version, and checksum.
- [x] Implement model validation script using CPU inference only.
- [x] Run focused test, process a short public video, and record speed and detection evidence.

## Day 3: Fire Model Selection

### Task 4: Compare public fire/smoke weights on fixed clips

**Files:** `scripts/validate_fire_model.py`, `docs/model_selection.md`, `assets/demo_videos/` (ignored), `worklog/daily_log.md`.

**Interfaces:** validation output records model source, supported classes, confidence threshold, positive/negative clip behavior, and CPU processing time.

- [x] Define comparison rows for flame, smoke, negative clips, load success, and CPU speed.
- [x] Obtain public weights and matching permitted demo clips; document licenses and attribution.
- [x] Run all candidates on the same clip set and input size.
- [x] Select one primary model; select a backup or narrow scope to flame-only if smoke results are unstable.
- [x] Save reproducible results and exact weight filenames.

## Day 4: Data Contracts and Model Registry

### Task 5: Implement SceneSpec, Detection, Event, and registry

**Files:** `src/schemas/*.py`, `src/models/base_adapter.py`, `src/models/model_registry.py`, `config/models.yaml`, `tests/test_scene_spec.py`, `tests/test_model_registry.py`.

**Interfaces:** `SceneSpec.model_validate(payload)`, `ModelRegistry.from_config(path)`, and `ModelRegistry.create(model_id)`; adapters return `list[Detection]`.

- [x] Write tests for valid scene config, unknown model, and unsupported rule rejection.
- [x] Run tests and confirm expected import/validation failures.
- [x] Implement minimal Pydantic models and adapter protocol.
- [x] Implement registry lookup and unknown-ID error.
- [x] Run both tests; verify valid config succeeds and invalid config is rejected.

## Day 5: Video Source and Writer

### Task 6: Read, sample, and write local video

**Files:** `src/video/video_source.py`, `src/video/frame_sampler.py`, `src/video/video_writer.py`, `tests/test_video_source.py`, `tests/test_frame_sampler.py`.

**Interfaces:** `VideoSource.open(path)`, `VideoSource.metadata()`, iterator yielding `(frame_id, timestamp_seconds, frame)`, `FrameSampler.should_process(frame_id)`, and `VideoWriter.write(frame)`.

- [x] Create tiny synthetic video fixtures in tests without committing binary video files.
- [x] Test metadata, unreadable file rejection, frame index, sampling cadence, and output readability.
- [x] Implement minimal OpenCV source, sampler, and writer.
- [x] Run focused tests and verify resources close on normal and exceptional exit.

## Day 6: Border Model Adapter and Tracking

### Task 7: Normalize general model output and track targets

**Files:** `src/models/yolo_adapter.py`, `src/tracking/tracker.py`, `src/visualization/annotator.py`, `tests/test_tracker.py`, `tests/test_annotator.py`.

**Interfaces:** `YoloAdapter.predict(frame) -> list[Detection]`; `Tracker.update(detections, timestamp_seconds) -> list[Detection]`; `Annotator.draw(frame, detections, scene_state) -> frame`.

- [x] Test normalized class, confidence, box, and stable IDs across adjacent frame fixtures.
- [x] Run tests and confirm they fail on missing modules/interfaces.
- [x] Implement CPU adapter and minimal short-term tracking.
- [x] Implement drawing for boxes, class labels, and IDs; keep confidence in data but omit it from the video overlay.
- [x] Run tests and process the selected border clip.

## Day 7: Border Event Rules

### Task 8: Evaluate forbidden regions, dwell, and cooldown

**Files:** `src/rules/base_rule.py`, `src/rules/region_rules.py`, `src/rules/temporal_rules.py`, `src/rules/rule_engine.py`, `tests/test_region_rules.py`, `tests/test_temporal_rules.py`.

**Interfaces:** `RuleEngine.evaluate(scene_spec, detections, frame_state) -> list[EventCandidate]`.

- [x] Write deterministic geometry and timestamp tests, including no-event boundaries.
- [x] Run tests and confirm expected failures.
- [x] Implement point-in-polygon, region transitions, dwell, and cooldown state.
- [x] Run focused tests for forbidden-region transitions, dwell, cooldown, and removed-rule rejection.

## Day 8: Fire Model and Consecutive Frames

### Task 9: Confirm fire alerts over consecutive frames

**Files:** `src/models/fire_adapter.py`, `src/rules/fire_rules.py`, `config/scenes/fire_detection.yaml`, `tests/test_fire_adapter.py`, `tests/test_fire_rules.py`.

**Interfaces:** `FireAdapter.predict(frame, frame_id) -> list[Detection]`; `ConsecutiveFramesRule.evaluate(detections, frame_state) -> list[EventCandidate]`.

- [x] Test insufficient frames, sufficient frames, brief gaps, reset, and cooldown.
- [x] Run tests and observe missing-module failures before implementation.
- [x] Implement shared YOLO output normalization and two-state suspected/confirmed rule.
- [x] Run tests and the first 100 frames of the selected fire clip; keep smoke performance marked unverified.

## Day 9: Event Storage and Exports

### Task 10: Persist screenshots, events, and run summary

**Files:** `src/events/event_manager.py`, `src/events/event_exporter.py`, `tests/test_event_manager.py`, `tests/test_event_exporter.py`.

**Interfaces:** `EventManager.add(candidate, frame) -> Event | None`; `EventExporter.export(run_dir, config, events, metrics) -> None`.

- [x] Test exact deduplication and snapshot creation using temporary directories; keep cooldown in `RuleEngine`.
- [x] Test JSON/CSV row equality and stable timestamp formatting.
- [x] Run tests and confirm failures due to missing implementation.
- [x] Implement event storage and exports.
- [x] Inspect generated JSON, CSV, summary, config, and JPEG files using a synthetic event/frame; short-video integration remains in Day 11.

## Day 10: DeepSeek Parser and Offline Templates

### Task 11: Convert user requests into allowlisted SceneSpec

**Files:** `src/llm/deepseek_client.py`, `src/llm/prompts.py`, `src/llm/scene_parser.py`, `config/scenes/*.yaml`, `tests/test_scene_parser.py`.

**Interfaces:** `DeepSeekClient.complete(messages) -> str`; `SceneParser.parse(text) -> SceneSpec`; `SceneParser.parse_or_template(text, template_id) -> SceneSpec`.

- [x] Write tests for valid JSON, invalid JSON, unknown model/rule, missing key, and offline fallback.
- [x] Run tests and confirm failure before implementation.
- [x] Implement API client with environment key, request timeout, one retry, and no key logging.
- [x] Implement restricted JSON parsing and Pydantic validation.
- [x] Add border-person intrusion, border-vehicle intrusion, border dwell, and fire templates.
- [x] Run tests without network using deterministic response fixtures; verify offline mode.

## Day 11: Analysis Pipeline

### Task 12: Connect source, adapter, tracking, rules, and exports

**Files:** `src/pipeline/analysis_pipeline.py`, `tests/test_pipeline_smoke.py`.

**Interfaces:** `AnalysisPipeline.run(video_path, scene_spec, run_dir, progress_callback=None, stop_event=None) -> RunSummary`.

- [x] Write pipeline test with a tiny generated clip and deterministic fake adapter.
- [x] Assert result video, summary, JSON, and CSV exist with matching event counts.
- [x] Run test and confirm it fails because the pipeline does not exist.
- [x] Implement a single orchestration path with resource cleanup and progress updates.
- [x] Run integration tests for both scene adapters on short clips derived from their fixed local videos.

## Day 12: Streamlit Interface

### Task 13: Connect upload, request, analysis, and download controls

**Files:** `app.py`, `src/ui/` if needed, `tests/test_app_smoke.py`, `README.md`.

**Interfaces:** UI submits validated `SceneSpec` to `AnalysisPipeline.run` and displays `RunSummary` without implementing model or rule logic.

- [x] Add AppTest assertions for upload label, scene input, template mode, and offline status.
- [x] Run tests and confirm expected missing UI controls.
- [x] Implement single-page controls, current model, progress, result video, events, and downloads.
- [x] Run AppTest and perform one manual end-to-end run for each scene.

## Day 13: Verification and Offline Package

### Task 14: Run acceptance tests and freeze functionality

**Files:** `docs/05_test_and_acceptance.md`, `worklog/daily_checklist.md`, `worklog/daily_log.md`.

- [x] Run all tests and record pass/fail output.
- [x] Process both fixed videos three times each; record duration and blockers.
- [x] Test missing API key using offline templates with zero network calls.
- [x] Generate stable backup output videos and check ignored asset/weight paths.
- [x] Fix only issues that block acceptance; do not add features.

## Day 14: User Documentation and Final Audit

### Task 15: Prepare detailed user instructions and delivery evidence

**Files:** `docs/03_feature_manual.md`, `docs/04_user_guide.md`, `docs/05_test_and_acceptance.md`, `README.md`, `worklog/daily_log.md`.

- [x] Document actual implemented functions and separate known limitations.
- [x] Write exact installation, launch, page-control, workflow, output, and fallback instructions.
- [x] Record six repeatable real-model runs and preserve the final outputs locally.
- [x] Verify private data, credentials, large weights, videos, and review intermediates are absent from Git.
- [x] Complete the final repository audit and freeze the Demo delivery branch.

## Plan Self-Review

- Spec coverage: model plugins, scene configuration, local video, rules, events, exports, UI, offline fallback, testing, daily logs, user documentation, and explicit non-goals all map to tasks above.
- Scope: camera/RTSP, model training, production database, local multimodal LLM, multi-stream processing, and full Agent Loop remain excluded.
- Interface consistency: all adapters return `list[Detection]`; the pipeline receives `SceneSpec` and writes a `RunSummary`; event exports consume normalized events.
- Dependency caution: add only imports required by a scheduled task; verify candidate weight compatibility and license before adoption.

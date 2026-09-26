# Final Demo Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a complete CPU-only two-scene demo, repeatable acceptance evidence, and a detailed Chinese user guide without defense slides, scripts, or reports.

**Architecture:** Preserve the frozen analysis pipeline and Streamlit page. Add only the missing evidence-view helper and a separate offline acceptance runner; keep validation logic outside the UI and use the existing model registry, templates, pipeline, and H.264 converter.

**Tech Stack:** Python 3.11, Streamlit 1.64, OpenCV, Ultralytics YOLO, Pydantic, PyYAML, imageio-ffmpeg, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-open-scene-vision-demo-design.md`, narrowed by the user on 2026-09-26 to exclude defense PPT, speech, rehearsal, and report deliverables.

## Global Constraints

- Run on the existing CPU-only Windows laptop.
- Accept local video files only; camera and RTSP remain excluded.
- DeepSeek is optional; offline templates must complete both demo flows without network access.
- Do not change model, rule, event, or pipeline contracts unless a failing acceptance test proves a blocking defect.
- Do not commit API keys, model weights, demo videos, generated runs, or personal paths.
- Add a failing test before every production behavior change.
- Final deliverables are runnable code, automated/real acceptance evidence, and detailed usage instructions.

---

### Task 1: Display event screenshots safely

**Files:**
- Modify: `src/ui/app_service.py`
- Modify: `src/ui/__init__.py`
- Modify: `app.py`
- Modify: `tests/test_ui_service.py`
- Modify: `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: event dictionaries containing optional relative `snapshot_path` values.
- Produces: `snapshot_paths_for_display(run_dir: str | Path, events: list[dict]) -> list[tuple[Path, dict]]`.

- [x] **Step 1: Write failing tests** proving valid snapshots are returned, traversal paths are rejected, and AppTest renders one event image.
- [x] **Step 2: Run focused tests** and confirm failure because the helper and page image do not exist.
- [x] **Step 3: Implement the minimal helper and screenshot gallery** using resolved-path containment checks and native `st.image`.
- [x] **Step 4: Run UI tests** and verify the event table, video, downloads, and screenshot remain visible.

### Task 2: Add repeatable offline acceptance runner

**Files:**
- Create: `src/validation/__init__.py`
- Create: `src/validation/acceptance.py`
- Create: `scripts/run_acceptance.py`
- Create: `tests/test_acceptance.py`

**Interfaces:**
- Consumes: fixed scenario definitions, local templates, model registry, local videos, and output root.
- Produces: `validate_run_artifacts(run_dir, expected_min_events) -> dict`, `run_acceptance(scenarios, repeat, output_root) -> dict`, and `runs/acceptance/acceptance_report.json`.

- [x] **Step 1: Write failing artifact-validation tests** for complete output, missing file, event-count mismatch, unreadable video, and non-H.264 result.
- [x] **Step 2: Run focused tests** and confirm failure because `src.validation` does not exist.
- [x] **Step 3: Implement artifact validation and report serialization** without loading model weights in unit tests.
- [x] **Step 4: Implement the CLI runner** with fixed `border_person_intrusion` and `fire_detection` templates, repeat count validation, progress output, H.264 conversion, and nonzero exit on failure.
- [x] **Step 5: Run focused tests** and verify deterministic fixtures pass.

### Task 3: Produce local demo clips, repeated evidence, and backups

**Files:**
- Create locally under ignored paths: `assets/demo_videos/border_demo.avi`, `assets/demo_videos/fire_demo.avi`
- Create locally under ignored paths: `runs/acceptance/`, `runs/backup/border/`, `runs/backup/fire/`
- Modify: `docs/05_test_and_acceptance.md`
- Modify: `worklog/daily_checklist.md`
- Modify: `worklog/daily_log.md`

**Interfaces:**
- Uses: `python scripts/run_acceptance.py --repeat 3`.
- Evidence: six successful real-model runs, event counts, elapsed time, H.264 video, screenshots, JSON/CSV consistency, and no network dependency.

- [x] **Step 1: Create 10–15 second local demo clips** from the already selected permitted source videos; do not commit binaries.
- [x] **Step 2: Run both scenes three times with `DEEPSEEK_API_KEY` absent** and save the machine-readable acceptance report.
- [x] **Step 3: Copy the last verified output for each scene into ignored backup directories** and open both H.264 videos to confirm readability.
- [x] **Step 4: Record exact commands, counts, timings, artifacts, and limitations** in the acceptance document and worklog.

### Task 4: Replace defense material with detailed user documentation

**Files:**
- Create: `docs/04_user_guide.md`
- Delete: `docs/06_defense_guide.md`
- Modify: `README.md`
- Modify: `docs/03_feature_manual.md`
- Modify: `docs/superpowers/plans/2026-09-17-open-scene-demo-implementation.md`
- Modify: `worklog/daily_checklist.md`

**Interfaces:** Documentation must match commands and UI text verified in the current repository.

- [x] **Step 1: Write installation and launch instructions** for PowerShell, virtual environment, dependencies, local weights, videos, and optional DeepSeek environment variable.
- [x] **Step 2: Document every page control and both complete workflows** with expected outputs and offline fallback behavior.
- [x] **Step 3: Document output files, troubleshooting, test commands, recovery steps, security rules, and known limitations** without unsupported claims.
- [x] **Step 4: Remove defense-only material and update repository links and remaining-stage wording** to reflect the user-approved deliverables.

### Task 5: Final completion audit

**Files:**
- Modify if evidence requires correction: `docs/03_feature_manual.md`, `docs/04_user_guide.md`, `docs/05_test_and_acceptance.md`, `README.md`, `worklog/daily_log.md`

**Interfaces:** Final gate consumes the complete repository and local acceptance outputs; it produces a clean feature branch ready for one final commit.

- [x] **Step 1: Run focused UI and acceptance tests.**
- [x] **Step 2: Run the full pytest suite, `pip check`, and `compileall`.**
- [x] **Step 3: Verify the Streamlit server starts and both backup H.264 videos are readable.**
- [x] **Step 4: Audit Git for secrets, weights, videos, run outputs, temporary files, whitespace errors, and stale defense/PPT references.**
- [x] **Step 5: Compare every final user requirement with direct evidence and leave no unverified completion claim.**

## Plan Self-Review

- Spec coverage: local video, DeepSeek/offline configuration, automatic model selection, border/fire processing, unified outputs, evidence screenshots, playback, downloads, repeated tests, and documentation are all covered.
- User scope: PPT, defense speech, rehearsal, and report deliverables are explicitly excluded and removed.
- Type consistency: the UI consumes event dictionaries already stored in `run_result`; acceptance uses existing `SceneSpec`, `AnalysisPipeline`, and `make_browser_playable` contracts.
- Placeholder scan: no TBD, TODO, deferred implementation, or undefined helper remains.

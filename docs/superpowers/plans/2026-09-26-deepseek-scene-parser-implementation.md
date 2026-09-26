# DeepSeek Scene Parser and Offline Templates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert a Chinese natural-language scene request into a locally validated `SceneSpec`, while retaining deterministic YAML templates when DeepSeek is unavailable or returns unsafe data.

**Architecture:** `DeepSeekClient` owns one JSON-mode HTTP call, environment-key lookup, timeout, and one transport retry. `SceneParser` owns prompts, JSON/Pydantic validation, model/target/rule compatibility checks, one invalid-output retry, and fixed-ID template fallback. The LLM never handles video frames and its output is never executed.

**Tech Stack:** Python 3.11, standard-library `urllib`, `json`, Pydantic, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-open-scene-vision-demo-design.md`

## Global Constraints

- Read the API key only from `DEEPSEEK_API_KEY` unless a test explicitly injects a fake key.
- Never log, return, serialize, or persist the API key.
- Use `https://api.deepseek.com/chat/completions` and JSON Output.
- Use `deepseek-flash` as the configurable default model.
- Retry one transport failure inside `DeepSeekClient`.
- Retry one invalid LLM output inside `SceneParser`.
- Accept only registered model IDs, supported target classes, implemented rule types, and allowlisted rule parameters.
- Template IDs map to fixed filenames; user text must never become a filesystem path.
- Unit tests must not call the network.
- Do not commit implementation until the user requests the next checkpoint.

---

### Task 1: DeepSeek HTTP Client

**Files:**
- Create: `src/llm/__init__.py`
- Create: `src/llm/deepseek_client.py`
- Create: `tests/test_deepseek_client.py`

**Interfaces:**
- Produces: `DeepSeekClient.complete(messages: Sequence[Mapping[str, str]]) -> str`.
- Produces: `MissingAPIKeyError`, `DeepSeekAPIError`.
- Accepts an injected `transport(request, timeout_seconds) -> bytes` for offline tests.

- [x] **Step 1: Write failing tests for a JSON-mode request and response extraction**

```python
def test_client_sends_json_mode_and_returns_message_content():
    captured = {}

    def transport(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        return json.dumps(
            {"choices": [{"message": {"content": '{"scene_type":"fire"}'}}]}
        ).encode("utf-8")

    client = DeepSeekClient(api_key="test-key", transport=transport)
    content = client.complete([{"role": "user", "content": "JSON please"}])

    assert json.loads(content) == {"scene_type": "fire"}
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["authorization"] == "Bearer test-key"
    assert captured["timeout"] == 30.0
```

- [x] **Step 2: Write failing tests for missing keys, one retry, and malformed responses**

```python
def test_client_rejects_missing_api_key_without_transport(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    calls = 0

    def transport(_request, _timeout):
        nonlocal calls
        calls += 1
        return b"{}"

    with pytest.raises(MissingAPIKeyError):
        DeepSeekClient(transport=transport).complete([])
    assert calls == 0


def test_client_retries_one_transport_failure():
    responses = [TimeoutError("slow"), RESPONSE_BYTES]

    def transport(_request, _timeout):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    assert DeepSeekClient(api_key="test-key", transport=transport).complete([]) == "{}"
    assert responses == []
```

- [x] **Step 3: Run the client tests and verify the missing-module failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_deepseek_client.py -q
```

- [x] **Step 4: Implement the minimal client**

Build a UTF-8 JSON request containing:

```python
{
    "model": self.model,
    "messages": list(messages),
    "response_format": {"type": "json_object"},
    "max_tokens": 1024,
    "stream": False,
}
```

Set `Authorization: Bearer <key>` and `Content-Type: application/json`. Retry exactly once for `TimeoutError`, `URLError`, and retryable HTTP/server failures. Reject empty or structurally invalid responses with `DeepSeekAPIError`.

- [x] **Step 5: Run Task 1 tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_deepseek_client.py -q
```

---

### Task 2: Safe Scene Parsing and Offline Templates

**Files:**
- Create: `src/llm/prompts.py`
- Create: `src/llm/scene_parser.py`
- Create: `config/scenes/border_person_intrusion.yaml`
- Create: `config/scenes/border_vehicle_intrusion.yaml`
- Create: `config/scenes/border_dwell.yaml`
- Test: `tests/test_scene_parser.py`

**Interfaces:**
- Produces: `build_scene_messages(user_text: str) -> list[dict[str, str]]`.
- Produces: `SceneParser.parse(text: str) -> SceneSpec`.
- Produces: `SceneParser.load_template(template_id: str) -> SceneSpec`.
- Produces: `SceneParser.parse_or_template(text: str, template_id: str) -> SceneSpec`.
- Produces: `SceneParseError`, `UnknownTemplateError`.

- [x] **Step 1: Write failing tests for valid JSON and one invalid-output retry**

```python
class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)

    def complete(self, _messages):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_parser_returns_validated_scene(registry, valid_border_json):
    scene = SceneParser(FakeClient(valid_border_json), registry).parse("有人进入禁区")
    assert scene.scene_type == "border"
    assert scene.model_id == "yolo_general"


def test_parser_retries_one_invalid_json_response(registry, valid_border_json):
    client = FakeClient("not json", valid_border_json)
    scene = SceneParser(client, registry).parse("有人进入禁区")
    assert scene.rules[0].type == "enter_region"
    assert client.responses == []
```

- [x] **Step 2: Write failing tests for unknown models, unsafe parameters, and compatibility**

Use literal JSON fixtures to prove these are rejected after two attempts:

- `model_id="unknown_model"`.
- `params={"command": "rm -rf"}`.
- `scene_type="fire"` with `model_id="yolo_general"`.
- target `airplane` for `yolo_general`.
- `object_present`, because the current `RuleEngine` does not implement it.

- [x] **Step 3: Write failing tests for fixed-ID templates and fallback**

```python
@pytest.mark.parametrize(
    "template_id",
    ["border_person_intrusion", "border_vehicle_intrusion", "border_dwell", "fire_detection"],
)
def test_all_offline_templates_are_valid(parser, template_id):
    assert parser.load_template(template_id).model_id in {"yolo_general", "fire_smoke"}


def test_missing_key_falls_back_to_selected_template(registry):
    parser = SceneParser(FakeClient(MissingAPIKeyError("missing")), registry)
    scene = parser.parse_or_template("火灾", "fire_detection")
    assert scene.scene_type == "fire"


def test_template_id_cannot_traverse_paths(parser):
    with pytest.raises(UnknownTemplateError):
        parser.load_template("../models")
```

- [x] **Step 4: Run parser tests and verify the missing-module failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_scene_parser.py -q
```

- [x] **Step 5: Implement the constrained prompt**

The system prompt must contain the word `JSON`, one complete output example, and these exact execution choices:

- Border: `scene_type=border`, `model_id=yolo_general`, targets from `person, car, motorcycle, bus, truck`, rules from `enter_region, leave_region, dwell`.
- Fire: `scene_type=fire`, `model_id=fire_smoke`, targets from `fire, smoke`, rule `consecutive_frames`.
- No code, commands, file paths, prose, or Markdown fences.

- [x] **Step 6: Implement parser validation**

After `json.loads` and `SceneSpec.model_validate`, enforce:

```python
MODEL_SCENES = {"yolo_general": "border", "fire_smoke": "fire"}
RULES_BY_SCENE = {
    "border": {"enter_region", "leave_region", "dwell"},
    "fire": {"consecutive_frames"},
}
PARAM_KEYS = {
    "enter_region": {"region_id", "polygon"},
    "leave_region": {"region_id", "polygon"},
    "dwell": {"region_id", "polygon", "seconds"},
    "consecutive_frames": {"rule_id", "frames", "max_gap_frames"},
}
```

Resolve the model through `ModelRegistry.get`, require every target to be in `ModelDefinition.classes`, reject extra parameter keys, validate polygon/numeric ranges, and allow only fixed template IDs.

- [x] **Step 7: Add three border templates**

Use a four-point 640x480 demonstration polygon. Person intrusion targets `person`; vehicle intrusion targets `car, bus, truck`; dwell targets `person` with `seconds: 5`. Keep the existing fire template unchanged.

- [x] **Step 8: Run Task 2 tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_scene_parser.py -q
```

---

### Task 3: Documentation and Verification

**Files:**
- Modify: `docs/03_feature_manual.md`
- Modify: `docs/superpowers/plans/2026-09-17-open-scene-demo-implementation.md`
- Modify: `worklog/daily_checklist.md`
- Modify: `worklog/daily_log.md`

- [x] **Step 1: Run focused tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_deepseek_client.py tests\test_scene_parser.py -q
```

- [x] **Step 2: Run the complete suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

- [x] **Step 3: Run offline acceptance without an API key**

Load all four templates, serialize each `SceneSpec`, and confirm no network transport is called.

- [x] **Step 4: Update status and work logs**

Record exact test counts. Mark API transport as implemented and offline-tested, but keep live DeepSeek account validation explicitly unverified until the user supplies a local key and requests a paid call.

- [ ] **Step 5: Stop at the next Git checkpoint**

Do not commit automatically. Report validation commands and recommend:

```text
feat(llm): add safe scene parsing
```

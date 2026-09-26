import json
from pathlib import Path

import pytest

from src.llm.deepseek_client import DeepSeekAPIError, MissingAPIKeyError
from src.llm.scene_parser import SceneParseError, SceneParser, UnknownTemplateError
from src.models.model_registry import ModelRegistry


TEMPLATE_DIR = Path("config/scenes")


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.received_messages = []

    def complete(self, messages):
        self.received_messages.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def registry():
    return ModelRegistry.from_config("config/models.yaml")


@pytest.fixture
def valid_border_payload():
    return {
        "scene_type": "border",
        "model_id": "yolo_general",
        "targets": ["person"],
        "rules": [
            {
                "type": "enter_region",
                "params": {
                    "region_id": "restricted_zone",
                    "polygon": [[100, 80], [540, 80], [540, 400], [100, 400]],
                },
            }
        ],
        "alert": {"enabled": True, "cooldown_seconds": 5},
    }


@pytest.fixture
def valid_border_json(valid_border_payload):
    return json.dumps(valid_border_payload, ensure_ascii=False)


def make_parser(client, registry):
    return SceneParser(client, registry, template_dir=TEMPLATE_DIR)


def test_parser_returns_validated_scene_and_sends_constrained_messages(
    registry,
    valid_border_json,
):
    client = FakeClient(valid_border_json)

    scene = make_parser(client, registry).parse("有人进入禁区后报警")

    assert scene.scene_type == "border"
    assert scene.model_id == "yolo_general"
    assert scene.rules[0].type == "enter_region"
    messages = client.received_messages[0]
    assert messages[0]["role"] == "system"
    assert "JSON" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "有人进入禁区后报警"}


def test_parser_retries_one_invalid_json_response(registry, valid_border_json):
    client = FakeClient("not json", valid_border_json)

    scene = make_parser(client, registry).parse("有人进入禁区")

    assert scene.rules[0].type == "enter_region"
    assert client.responses == []


def test_parser_stops_after_two_invalid_outputs(registry):
    client = FakeClient("not json", "still not json")

    with pytest.raises(SceneParseError, match="two invalid"):
        make_parser(client, registry).parse("有人进入禁区")

    assert client.responses == []


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ({"model_id": "unknown_model"}, "unknown model"),
        ({"scene_type": "fire"}, "does not belong"),
        ({"targets": ["airplane"]}, "unsupported targets"),
    ],
)
def test_parser_rejects_model_and_target_mismatches(
    registry,
    valid_border_payload,
    mutation,
    expected_message,
):
    invalid = {**valid_border_payload, **mutation}
    response = json.dumps(invalid)

    with pytest.raises(SceneParseError, match=expected_message):
        make_parser(FakeClient(response, response), registry).parse("test")


def test_parser_rejects_unsafe_rule_parameters(registry, valid_border_payload):
    invalid = {
        **valid_border_payload,
        "rules": [
            {
                "type": "enter_region",
                "params": {
                    "region_id": "restricted_zone",
                    "polygon": [[0, 0], [10, 0], [10, 10]],
                    "command": "rm -rf /",
                },
            }
        ],
    }
    response = json.dumps(invalid)

    with pytest.raises(SceneParseError, match="unsupported parameters"):
        make_parser(FakeClient(response, response), registry).parse("test")


def test_parser_rejects_schema_rule_not_implemented_by_engine(
    registry,
    valid_border_payload,
):
    invalid = {
        **valid_border_payload,
        "rules": [{"type": "object_present", "params": {}}],
    }
    response = json.dumps(invalid)

    with pytest.raises(SceneParseError, match="not supported for border"):
        make_parser(FakeClient(response, response), registry).parse("test")


def test_parser_rejects_invalid_polygon_and_numeric_ranges(
    registry,
    valid_border_payload,
):
    bad_polygon = {
        **valid_border_payload,
        "rules": [
            {
                "type": "dwell",
                "params": {
                    "region_id": "zone",
                    "polygon": [[0, 0], [10, 0]],
                    "seconds": -1,
                },
            }
        ],
    }
    response = json.dumps(bad_polygon)

    with pytest.raises(SceneParseError, match="polygon"):
        make_parser(FakeClient(response, response), registry).parse("test")


def test_parser_rejects_invalid_dwell_seconds(registry, valid_border_payload):
    invalid = {
        **valid_border_payload,
        "rules": [
            {
                "type": "dwell",
                "params": {
                    "region_id": "zone",
                    "polygon": [[0, 0], [10, 0], [10, 10]],
                    "seconds": 0,
                },
            }
        ],
    }
    response = json.dumps(invalid)

    with pytest.raises(SceneParseError, match="seconds"):
        make_parser(FakeClient(response, response), registry).parse("test")


def test_parser_rejects_invalid_consecutive_frame_count(registry):
    invalid = {
        "scene_type": "fire",
        "model_id": "fire_smoke",
        "targets": ["fire"],
        "rules": [
            {
                "type": "consecutive_frames",
                "params": {"rule_id": "fire_sequence", "frames": 0},
            }
        ],
    }
    response = json.dumps(invalid)

    with pytest.raises(SceneParseError, match="frames"):
        make_parser(FakeClient(response, response), registry).parse("test")


@pytest.mark.parametrize(
    "mutation",
    [
        {"rules": [{"type": "run_shell", "params": {}}]},
        {"python_code": "print('unsafe')"},
    ],
)
def test_parser_rejects_unknown_rules_and_extra_top_level_fields(
    registry,
    valid_border_payload,
    mutation,
):
    invalid = {**valid_border_payload, **mutation}
    response = json.dumps(invalid)

    with pytest.raises(SceneParseError, match="two invalid"):
        make_parser(FakeClient(response, response), registry).parse("test")


@pytest.mark.parametrize(
    ("template_id", "scene_type", "model_id"),
    [
        ("border_person_intrusion", "border", "yolo_general"),
        ("border_vehicle_intrusion", "border", "yolo_general"),
        ("border_dwell", "border", "yolo_general"),
        ("fire_detection", "fire", "fire_smoke"),
    ],
)
def test_all_offline_templates_are_valid(
    registry,
    template_id,
    scene_type,
    model_id,
):
    parser = make_parser(FakeClient(), registry)

    scene = parser.load_template(template_id)

    assert scene.scene_type == scene_type
    assert scene.model_id == model_id


def test_missing_key_falls_back_to_selected_template(registry):
    parser = make_parser(
        FakeClient(MissingAPIKeyError("missing")),
        registry,
    )

    scene = parser.parse_or_template("检测火灾", "fire_detection")

    assert scene.scene_type == "fire"
    assert scene.model_id == "fire_smoke"


def test_api_failure_falls_back_to_selected_template(registry):
    parser = make_parser(
        FakeClient(DeepSeekAPIError("offline")),
        registry,
    )

    scene = parser.parse_or_template("检测禁区", "border_person_intrusion")

    assert scene.scene_type == "border"


def test_invalid_output_falls_back_after_one_retry(registry):
    parser = make_parser(FakeClient("bad", "bad again"), registry)

    scene = parser.parse_or_template("检测火灾", "fire_detection")

    assert scene.scene_type == "fire"


def test_template_id_cannot_traverse_paths(registry):
    parser = make_parser(FakeClient(), registry)

    with pytest.raises(UnknownTemplateError, match="unknown template"):
        parser.load_template("../models")

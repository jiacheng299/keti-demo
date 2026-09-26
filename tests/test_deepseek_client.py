import json
from urllib.error import HTTPError

import pytest

from src.llm.deepseek_client import (
    DeepSeekAPIError,
    DeepSeekClient,
    MissingAPIKeyError,
)


RESPONSE_BYTES = json.dumps(
    {"choices": [{"message": {"content": "{}"}}]}
).encode("utf-8")


def test_client_sends_json_mode_and_returns_message_content():
    captured = {}

    def transport(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers["Authorization"]
        captured["content_type"] = request.headers["Content-type"]
        captured["timeout"] = timeout
        captured["url"] = request.full_url
        return json.dumps(
            {"choices": [{"message": {"content": '{"scene_type":"fire"}'}}]}
        ).encode("utf-8")

    client = DeepSeekClient(api_key="test-key", transport=transport)
    content = client.complete([{"role": "user", "content": "JSON please"}])

    assert json.loads(content) == {"scene_type": "fire"}
    assert captured["body"] == {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "JSON please"}],
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,
        "stream": False,
    }
    assert captured["authorization"] == "Bearer test-key"
    assert captured["content_type"] == "application/json"
    assert captured["timeout"] == 30.0
    assert captured["url"] == "https://api.deepseek.com/chat/completions"


def test_client_rejects_missing_api_key_without_calling_transport(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    calls = 0

    def transport(_request, _timeout):
        nonlocal calls
        calls += 1
        return RESPONSE_BYTES

    with pytest.raises(MissingAPIKeyError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient(transport=transport).complete([])

    assert calls == 0


def test_client_reads_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-test-key")
    authorization = None

    def transport(request, _timeout):
        nonlocal authorization
        authorization = request.headers["Authorization"]
        return RESPONSE_BYTES

    DeepSeekClient(transport=transport).complete([])

    assert authorization == "Bearer environment-test-key"


def test_client_retries_one_transport_failure():
    responses = [TimeoutError("slow"), RESPONSE_BYTES]

    def transport(_request, _timeout):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    content = DeepSeekClient(api_key="test-key", transport=transport).complete([])

    assert content == "{}"
    assert responses == []


def test_client_retries_retryable_http_error():
    responses = [HTTPError("url", 503, "busy", {}, None), RESPONSE_BYTES]

    def transport(_request, _timeout):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    content = DeepSeekClient(api_key="test-key", transport=transport).complete([])

    assert content == "{}"
    assert responses == []


def test_client_does_not_retry_authentication_error():
    calls = 0

    def transport(_request, _timeout):
        nonlocal calls
        calls += 1
        raise HTTPError("url", 401, "unauthorized", {}, None)

    with pytest.raises(DeepSeekAPIError, match="401"):
        DeepSeekClient(api_key="test-key", transport=transport).complete([])

    assert calls == 1


@pytest.mark.parametrize(
    "response",
    [
        b"not json",
        b"{}",
        json.dumps({"choices": [{"message": {"content": ""}}]}).encode("utf-8"),
    ],
)
def test_client_rejects_malformed_or_empty_responses(response):
    client = DeepSeekClient(
        api_key="test-key",
        transport=lambda _request, _timeout: response,
    )

    with pytest.raises(DeepSeekAPIError, match="response"):
        client.complete([])

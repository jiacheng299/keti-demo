"""Minimal DeepSeek JSON-mode client with bounded retry behavior."""

from collections.abc import Callable, Mapping, Sequence
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class MissingAPIKeyError(RuntimeError):
    """Raised before networking when no DeepSeek API key is configured."""


class DeepSeekAPIError(RuntimeError):
    """Raised when the DeepSeek request or response cannot be used."""

    def __init__(
        self, message: str, *, status_code: int | None = None,
        error_code: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


Transport = Callable[[Request, float], bytes]


def _urlopen_transport(request: Request, timeout_seconds: float) -> bytes:
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


class DeepSeekClient:
    """Call DeepSeek Chat Completions without exposing the API key."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "deepseek-flash",
        endpoint: str = "https://api.deepseek.com/chat/completions",
        timeout_seconds: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        self._api_key = (api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY", "")).strip()
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self._transport = transport or _urlopen_transport

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return assistant JSON text after at most two transport attempts."""
        if not self._api_key:
            raise MissingAPIKeyError("DEEPSEEK_API_KEY is not configured")

        request = self._build_request(messages)
        response_bytes = self._send_with_retry(request)
        return self._extract_content(response_bytes)

    def _build_request(self, messages: Sequence[Mapping[str, str]]) -> Request:
        body = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "response_format": {"type": "json_object"},
            # Scene extraction only needs final JSON. Thinking otherwise shares
            # the output budget and can leave no tokens for the actual config.
            "thinking": {"type": "disabled"},
            "max_tokens": 4096,
            "stream": False,
        }
        return Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

    def _send_with_retry(self, request: Request) -> bytes:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return self._transport(request, self.timeout_seconds)
            except HTTPError as error:
                if error.code not in {408, 409, 429} and error.code < 500:
                    raise DeepSeekAPIError(
                        f"DeepSeek request failed with HTTP {error.code}",
                        status_code=error.code,
                        error_code="http_error",
                    ) from error
                last_error = error
            except (TimeoutError, URLError, OSError) as error:
                last_error = error

            if attempt == 1:
                break

        raise DeepSeekAPIError(
            "DeepSeek request failed after one retry",
            status_code=last_error.code if isinstance(last_error, HTTPError) else None,
            error_code=(
                "http_error" if isinstance(last_error, HTTPError)
                else "timeout" if isinstance(last_error, TimeoutError)
                or isinstance(getattr(last_error, "reason", None), TimeoutError)
                else "connection_error"
            ),
        ) from last_error

    @staticmethod
    def _extract_content(response_bytes: bytes) -> str:
        try:
            payload: Any = json.loads(response_bytes.decode("utf-8"))
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise DeepSeekAPIError(
                "DeepSeek response has an invalid structure", error_code="invalid_response"
            ) from error

        if finish_reason == "length":
            raise DeepSeekAPIError(
                "DeepSeek response exceeded its output limit", error_code="output_truncated"
            )
        if finish_reason == "content_filter":
            raise DeepSeekAPIError(
                "DeepSeek response was filtered", error_code="content_filtered"
            )

        if not isinstance(content, str) or not content.strip():
            raise DeepSeekAPIError(
                "DeepSeek response content is empty", error_code="empty_response"
            )
        return content

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
        timeout_seconds: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
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
            "max_tokens": 1024,
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
                        f"DeepSeek request failed with HTTP {error.code}"
                    ) from error
                last_error = error
            except (TimeoutError, URLError, OSError) as error:
                last_error = error

            if attempt == 1:
                break

        raise DeepSeekAPIError("DeepSeek request failed after one retry") from last_error

    @staticmethod
    def _extract_content(response_bytes: bytes) -> str:
        try:
            payload: Any = json.loads(response_bytes.decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise DeepSeekAPIError("DeepSeek response has an invalid structure") from error

        if not isinstance(content, str) or not content.strip():
            raise DeepSeekAPIError("DeepSeek response content is empty")
        return content

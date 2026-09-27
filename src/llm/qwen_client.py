"""Bounded visual API calls; credentials and thinking traces never enter reports."""
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.security.credential_store import load_api_key

SETTINGS_PATH = Path(__file__).resolve().parents[2] / "runs/settings/qwen.json"


class QwenError(ValueError):
    pass


class QwenSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = "qwen3-vl-32b-thinking"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    daily_requests: int = Field(default=40, ge=1, le=200)
    public_enabled: bool = True

    @field_validator("model")
    @classmethod
    def validate_model(cls, value):
        if not re.fullmatch(r"qwen[a-zA-Z0-9.-]*vl[a-zA-Z0-9.-]*", value):
            raise ValueError("请选择 Qwen 视觉模型名称")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value):
        value = value.strip().rstrip("/")
        parts = urlsplit(value)
        known = {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"}
        workspace = re.fullmatch(r"[a-z0-9-]+\.(cn-beijing|ap-southeast-1|ap-northeast-1|us-east-1|eu-central-1|cn-hongkong)\.maas\.aliyuncs\.com", parts.hostname or "")
        if (parts.scheme != "https" or parts.username or parts.password or parts.port
                or parts.query or parts.fragment or parts.path != "/compatible-mode/v1"
                or (parts.hostname not in known and not workspace)):
            raise ValueError("请填写百炼官方 HTTPS Base URL，以 /compatible-mode/v1 结尾")
        return value


def load_qwen_settings(path=None):
    target = Path(path or SETTINGS_PATH)
    if not target.exists():
        return QwenSettings()
    try:
        return QwenSettings.model_validate_json(target.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise QwenError("Qwen 设置文件不可用，请在本机页面重新保存设置。") from None


def save_qwen_settings(settings, path=None):
    target = Path(path or SETTINGS_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(settings.model_dump_json(indent=2))
        os.replace(temporary, target)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _transport(request, timeout):
    with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
        body = response.read(2_000_001)
        if len(body) > 2_000_000:
            raise QwenError("Qwen 响应过长，本次检验未完成。")
        return body


class QwenClient:
    def __init__(self, settings=None, api_key=None, transport=None):
        self.settings = settings or load_qwen_settings()
        self._key = api_key if api_key is not None else (load_api_key("qwen") or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY", ""))
        self._transport = transport or _transport

    def complete(self, content, *, max_tokens=4096):
        if not self._key:
            raise QwenError("未配置 Qwen API Key，请在本机运行状态中保存。")
        request = Request(self.settings.base_url + "/chat/completions", data=json.dumps({
            "model": self.settings.model, "messages": [{"role": "user", "content": content}],
            "stream": False, "max_tokens": max_tokens,
        }).encode("utf-8"), headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"})
        try:
            payload = json.loads(self._transport(request, 90))
        except HTTPError as error:
            reason = {400: "请求参数或输入格式不被该模型支持", 401: "Key 无效或与地域不匹配", 403: "没有模型权限", 404: "模型或接口地址不存在", 429: "请求达到限额，请稍后重试"}.get(error.code, "服务暂时不可用")
            raise QwenError(f"Qwen 调用失败（HTTP {error.code}）：{reason}。") from None
        except (TimeoutError, URLError):
            raise QwenError("Qwen 连接失败或等待超过90秒。本次没有自动重试，可稍后重试。") from None
        except (ValueError, OSError):
            raise QwenError("Qwen 返回内容异常，本次检验未完成。") from None
        try:
            choice = payload["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise QwenError("Qwen 未返回完整答案（可能达到输出上限），本次检验未完成。")
            answer = choice["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise QwenError("Qwen 没有返回最终答案，本次检验未完成。")
            # Never expose or persist reasoning_content.
            return answer, payload.get("usage", {})
        except (KeyError, IndexError, TypeError):
            raise QwenError("Qwen 返回结构异常，本次检验未完成。") from None

    def test_connection(self):
        self.complete("Reply only OK.", max_tokens=512)
        return "连接成功：所选模型已返回最终回答。此测试未上传视频。"

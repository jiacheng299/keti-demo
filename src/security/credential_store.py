"""Keep API keys in the operating system credential manager."""

import keyring

SERVICE = "keti-demo-deepseek"
ACCOUNT = "api-key"
SERVICES = {"deepseek": SERVICE, "qwen": "keti-demo-qwen"}


class CredentialStoreError(RuntimeError):
    """A safe, user-facing credential error without backend details."""


def _service(provider: str) -> str:
    if provider not in SERVICES:
        raise CredentialStoreError("不支持的 API 服务。")
    return SERVICES[provider]


def load_api_key(provider: str = "deepseek") -> str | None:
    service = _service(provider)
    try:
        return keyring.get_password(service, ACCOUNT) or None
    except Exception:
        raise CredentialStoreError("无法读取系统凭据管理器，可使用环境变量或离线模板。") from None


def save_api_key(api_key: str, provider: str = "deepseek") -> None:
    service = _service(provider)
    value = api_key.strip()
    if not value:
        raise CredentialStoreError("请先输入 API Key。")
    if any(character.isspace() for character in value) or not value.isascii():
        raise CredentialStoreError("API Key 不能包含空白或非英文字符，请检查复制内容。")
    try:
        keyring.set_password(service, ACCOUNT, value)
    except Exception:
        raise CredentialStoreError("保存失败：系统凭据管理器不可用，请重试。") from None


def delete_api_key(provider: str = "deepseek") -> None:
    service = _service(provider)
    try:
        if keyring.get_password(service, ACCOUNT) is not None:
            keyring.delete_password(service, ACCOUNT)
    except Exception:
        raise CredentialStoreError("清除失败：无法访问系统凭据管理器，请重试。") from None

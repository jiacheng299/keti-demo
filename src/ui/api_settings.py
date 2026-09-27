"""Independent, persistent provider credentials with local-only editing."""

import os

import streamlit as st
from src.ui.share_access import is_shared_instance
from src.security.credential_store import (
    CredentialStoreError, delete_api_key, load_api_key, save_api_key,
)
from src.llm.qwen_client import QwenClient, QwenSettings, QwenError, load_qwen_settings, save_qwen_settings
from src.security.qwen_budget import QwenBudget


PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek", "environment": ("DEEPSEEK_API_KEY",),
        "purpose": "用于场景需求解析。",
    },
    "qwen": {
        "name": "Qwen", "environment": ("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
        "purpose": "用于本地模型不支持时的云端视频抽帧检验。",
    },
}


def _save_key(provider: str = "deepseek") -> None:
    if is_shared_instance():
        return
    name = PROVIDERS[provider]["name"]
    try:
        save_api_key(st.session_state.get(f"{provider}_key_input", ""), provider)
    except CredentialStoreError as error:
        st.session_state[f"{provider}_settings_message"] = ("error", str(error))
    else:
        st.session_state[f"{provider}_settings_message"] = (
            "success", f"{name} API Key 已保存到系统凭据管理器；如已有密钥，已替换。保存不代表接口验证通过。"
        )
    finally:
        st.session_state[f"{provider}_key_input"] = ""


def _clear_key(provider: str = "deepseek") -> None:
    if is_shared_instance():
        return
    try:
        delete_api_key(provider)
    except CredentialStoreError as error:
        st.session_state[f"{provider}_settings_message"] = ("error", str(error))
    else:
        st.session_state[f"{provider}_key_input"] = ""
        st.session_state[f"{provider}_settings_message"] = (
            "success", f"已清除保存的 {PROVIDERS[provider]['name']} API Key。"
        )


def _render_provider(provider: str) -> str | None:
    config = PROVIDERS[provider]
    name = config["name"]
    credential_error = None
    try:
        stored_key = load_api_key(provider)
    except CredentialStoreError as error:
        stored_key = None
        credential_error = str(error)
    environment_key = next((os.environ[key].strip() for key in config["environment"]
                            if os.getenv(key, "").strip()), None)
    if stored_key:
        st.success(f"{name} API：已配置（系统凭据）")
    elif environment_key:
        st.success(f"{name} API：已配置（环境变量）")
    elif provider == "deepseek":
        st.warning("DeepSeek API：未配置，可使用离线模板")
    else:
        st.info("Qwen API：未配置，不影响现有本地视频分析")

    if is_shared_instance():
        return stored_key or environment_key

    with st.expander(f"{name} API 设置", expanded=not (stored_key or environment_key)):
        st.caption(config["purpose"])
        st.text_input(
            f"{name} API Key", type="password", key=f"{provider}_key_input",
            help="输入新 Key 后点击保存即可替换旧 Key。已保存的密钥不会回填到输入框。",
        )
        with st.container(horizontal=True):
            st.button("保存 / 修改 API Key", key="save_api_key" if provider == "deepseek" else "save_qwen_api_key",
                      on_click=_save_key, args=(provider,))
            st.button("清除已保存 Key", key="clear_api_key" if provider == "deepseek" else "clear_qwen_api_key",
                      on_click=_clear_key, args=(provider,))
        st.caption("保存到本机系统凭据管理器，重启后仍可使用。输入框留空不会删除原密钥；保存不会调用接口。")
        if environment_key:
            st.caption("已保存密钥优先于环境变量。清除按钮只删除保存的密钥，启动环境中的密钥仍可生效。")
        if credential_error:
            st.error(credential_error)
        message = st.session_state.pop(f"{provider}_settings_message", None)
        if message:
            level, content = message
            getattr(st, level)(content)
        if provider == "qwen":
            render_qwen_options()
    return stored_key or environment_key


def render_qwen_options():
    try:
        settings = load_qwen_settings()
    except QwenError as error:
        st.error(str(error))
        settings = QwenSettings()
    with st.form("qwen_options"):
        model = st.text_input("Qwen 模型名称", value=settings.model)
        url = st.text_input("Qwen Base URL", value=settings.base_url)
        limit = st.number_input("每日请求上限（本机与公网共用）", min_value=1, max_value=200, value=settings.daily_requests)
        public = st.checkbox("允许公网访客使用 Qwen 检验", value=settings.public_enabled)
        saved = st.form_submit_button("保存模型与调用设置")
    if saved:
        try:
            settings = QwenSettings(model=model.strip(), base_url=url, daily_requests=limit, public_enabled=public)
            save_qwen_settings(settings)
            st.success("模型与调用设置已保存，重启后仍生效。")
        except (ValueError, OSError):
            st.error("保存失败：请检查视觉模型名称、百炼官方 Base URL 和目录写入权限。")
    st.caption(f"当前模型：{settings.model}；今日已尝试 {QwenBudget(settings.daily_requests).used()}/{settings.daily_requests} 次请求（失败也计数）。")
    st.caption("修改模型或地址后先保存。测试连接只发送简短文字，不上传视频，也计入请求额度。")
    if st.button("测试 Qwen 连接", key="test_qwen_connection"):
        try:
            with st.spinner("正在测试 Qwen……"):
                with QwenBudget(settings.daily_requests).session() as budget:
                    budget.reserve()
                    message = QwenClient(settings=settings).test_connection()
            st.success(message)
        except (QwenError, CredentialStoreError) as error:
            st.error(str(error))


def render_api_settings() -> str | None:
    """Render both providers; preserve the current DeepSeek caller's return type."""
    deepseek_key = _render_provider("deepseek")
    _render_provider("qwen")
    if is_shared_instance():
        st.caption("分享模式：API 由分享者在本机页面配置，访客不能修改或删除本机密钥。")
    return deepseek_key

"""Tests must never access the user's actual operating-system credentials."""
import keyring
import pytest


@pytest.fixture(autouse=True)
def fake_credential_backend(monkeypatch):
    values = {}
    monkeypatch.setattr(keyring, "get_password", lambda service, account: values.get((service, account)))
    monkeypatch.setattr(keyring, "set_password", lambda service, account, value: values.__setitem__((service, account), value))
    monkeypatch.setattr(keyring, "delete_password", lambda service, account: values.pop((service, account)))
    return values


@pytest.fixture(autouse=True)
def isolated_qwen_settings(monkeypatch, tmp_path):
    monkeypatch.setattr('src.llm.qwen_client.SETTINGS_PATH', tmp_path/'qwen.json')
    monkeypatch.setattr('src.security.qwen_budget.DEFAULT_DB', tmp_path/'budget.sqlite')

import keyring
import pytest

from src.security.credential_store import (
    CredentialStoreError, delete_api_key, load_api_key, save_api_key,
)


def test_credential_round_trip_and_idempotent_delete():
    assert load_api_key() is None
    save_api_key("  test-only-key  ")
    assert load_api_key() == "test-only-key"
    delete_api_key()
    delete_api_key()
    assert load_api_key() is None


def test_provider_credentials_are_independent_and_replacement_persists():
    save_api_key("deepseek-original")
    save_api_key("qwen-original", "qwen")
    save_api_key("qwen-replacement", "qwen")
    assert load_api_key() == "deepseek-original"
    assert load_api_key("qwen") == "qwen-replacement"
    delete_api_key("qwen")
    delete_api_key("qwen")
    assert load_api_key("qwen") is None
    assert load_api_key() == "deepseek-original"


def test_unknown_provider_cannot_write_to_an_arbitrary_service(fake_credential_backend):
    with pytest.raises(CredentialStoreError):
        save_api_key("test-only-key", "unknown")
    assert not fake_credential_backend


@pytest.mark.parametrize("value", ["", "  ", "bad key", "密钥"])
def test_invalid_key_does_not_replace_saved_credential(value):
    save_api_key("existing-test-key")
    with pytest.raises(CredentialStoreError):
        save_api_key(value)
    assert load_api_key() == "existing-test-key"


@pytest.mark.parametrize("operation,backend_method", [(load_api_key, "get_password"), (lambda: save_api_key("test-only-key"), "set_password"), (delete_api_key, "delete_password")])
def test_backend_error_never_exposes_credentials(monkeypatch, operation, backend_method):
    save_api_key("test-only-key")

    def fail(*args):
        raise RuntimeError("private-backend-details test-only-key")

    monkeypatch.setattr(keyring, backend_method, fail)
    with pytest.raises(CredentialStoreError) as caught:
        operation()
    assert "test-only-key" not in str(caught.value)
    assert "private-backend-details" not in str(caught.value)

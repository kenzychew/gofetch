"""Tests for the demo/app.py Application Default Credentials shim.

Uses only fake, non-shaped placeholder JSON -- never real key material.
"""

import os
import stat

import pytest

from demo.app import _materialize_adc_from_env

_FAKE_KEY_JSON = '{"type": "service_account", "project_id": "fake-project"}'


def test_unset_env_falls_through_to_adc_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """No GOOGLE_APPLICATION_CREDENTIALS_JSON means no change to ADC env."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    _materialize_adc_from_env()

    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_blank_env_falls_through_to_adc_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank/whitespace value is treated the same as unset."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "   ")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    _materialize_adc_from_env()

    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_valid_json_materializes_restricted_temp_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid JSON blob is written to a 0600 temp file outside the repo."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", _FAKE_KEY_JSON)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    _materialize_adc_from_env()

    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    assert cred_path is not None
    try:
        assert os.path.isfile(cred_path)
        with open(cred_path) as f:
            assert f.read() == _FAKE_KEY_JSON
        mode = stat.S_IMODE(os.stat(cred_path).st_mode)
        assert mode == 0o600
    finally:
        os.unlink(cred_path)


def test_invalid_json_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed JSON fails loudly at startup instead of a silent fallback."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "not valid json")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    with pytest.raises(RuntimeError, match="not valid JSON"):
        _materialize_adc_from_env()

    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from rpi_agent.config import resolve_config, resolve_state_dir
from rpi_agent.constants import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_MODEL
from rpi_agent.exceptions import ConfigurationError

ENV_VARS = (
    "STARK_API_KEY",
    "OPENAI_API_KEY",
    "STARK_BASE_URL",
    "OPENAI_BASE_URL",
    "STARK_MODEL",
    "OPENAI_MODEL",
    "RPI_AGENT_MAX_RETRIES",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def make_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "api_key": None,
        "base_url": None,
        "model": None,
        "skills_dir": Path("~/skills"),
        "yolo": False,
        "no_prune": False,
        "max_retries": None,
        "resume": None,
        "test_discovery": False,
        "test_tools": False,
        "test_agent": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_missing_api_key_raises() -> None:
    with pytest.raises(ConfigurationError, match="Missing API Key"):
        resolve_config(make_args())


def test_defaults_applied_when_only_key_given() -> None:
    config = resolve_config(make_args(api_key="cli-key"))
    assert config.api_key == "cli-key"
    assert config.base_url == DEFAULT_BASE_URL
    assert config.model_name == DEFAULT_MODEL
    assert config.prune_context is True
    assert config.yolo is False


def test_skills_dir_is_expanded_and_resolved() -> None:
    config = resolve_config(make_args(api_key="k"))
    assert config.skills_dir == (Path.home() / "skills").resolve()


def test_stark_env_takes_precedence_over_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STARK_API_KEY", "stark-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("STARK_BASE_URL", "http://stark")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai")
    monkeypatch.setenv("STARK_MODEL", "stark-model")
    monkeypatch.setenv("OPENAI_MODEL", "openai-model")

    config = resolve_config(make_args())
    assert (config.api_key, config.base_url, config.model_name) == (
        "stark-key",
        "http://stark",
        "stark-model",
    )


def test_openai_env_used_as_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai")
    monkeypatch.setenv("OPENAI_MODEL", "openai-model")

    config = resolve_config(make_args())
    assert (config.api_key, config.base_url, config.model_name) == (
        "openai-key",
        "http://openai",
        "openai-model",
    )


def test_cli_args_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STARK_API_KEY", "env-key")
    monkeypatch.setenv("STARK_BASE_URL", "http://env")
    monkeypatch.setenv("STARK_MODEL", "env-model")

    config = resolve_config(
        make_args(api_key="cli-key", base_url="http://cli", model="cli-model")
    )
    assert (config.api_key, config.base_url, config.model_name) == (
        "cli-key",
        "http://cli",
        "cli-model",
    )


def test_flags_are_mapped() -> None:
    config = resolve_config(
        make_args(
            api_key="k",
            yolo=True,
            no_prune=True,
            test_discovery=True,
            test_tools=True,
            test_agent=True,
        )
    )
    assert config.yolo is True
    assert config.prune_context is False
    assert config.test_discovery and config.test_tools and config.test_agent


def test_max_retries_default() -> None:
    assert resolve_config(make_args(api_key="k")).max_retries == DEFAULT_MAX_RETRIES


def test_max_retries_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RPI_AGENT_MAX_RETRIES", "9")
    assert resolve_config(make_args(api_key="k")).max_retries == 9


def test_max_retries_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RPI_AGENT_MAX_RETRIES", "9")
    assert resolve_config(make_args(api_key="k", max_retries=0)).max_retries == 0


def test_max_retries_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RPI_AGENT_MAX_RETRIES", "lots")
    with pytest.raises(ConfigurationError, match="must be an integer, got 'lots'"):
        resolve_config(make_args(api_key="k"))


def test_max_retries_negative() -> None:
    with pytest.raises(ConfigurationError, match="must be >= 0"):
        resolve_config(make_args(api_key="k", max_retries=-1))


def test_resume_is_mapped() -> None:
    assert resolve_config(make_args(api_key="k")).resume is None
    assert resolve_config(make_args(api_key="k", resume="latest")).resume == "latest"


def test_state_dir_is_repo_scoped(tmp_path: Path) -> None:
    path = resolve_state_dir(tmp_path)
    assert path == tmp_path / ".rpi_agent"
    assert path.is_dir()


def test_state_dir_falls_back_to_home_when_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    repo = tmp_path / "repo"
    repo.mkdir()
    # A regular file where the directory should be makes mkdir fail.
    (repo / ".rpi_agent").write_text("not a dir")

    path = resolve_state_dir(repo)
    assert path == home / ".rpi_agent" / "repo"
    assert path.is_dir()

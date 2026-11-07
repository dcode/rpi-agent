from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from rpi_agent.config import load_user_config, resolve_config, resolve_state_dir
from rpi_agent.constants import (
    DEFAULT_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_SKILLS_DIR,
    GEMINI_BASE_URL,
)
from rpi_agent.exceptions import ConfigurationError

ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "RPI_AGENT_MAX_RETRIES",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


def make_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "config": None,
        "api_key": None,
        "base_url": None,
        "gemini": False,
        "model": None,
        "skills_dir": DEFAULT_SKILLS_DIR,
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


def write_default_config(tmp_path: Path, content: str) -> Path:
    cfg = tmp_path / ".config" / "rpi-agent.yml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(content, encoding="utf-8")
    return cfg


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


def test_gemini_flag_sets_gemini_endpoint_and_default_model() -> None:
    config = resolve_config(make_args(api_key="cli-key", gemini=True))
    assert config.base_url == GEMINI_BASE_URL
    assert config.model_name == DEFAULT_GEMINI_MODEL


def test_gemini_flag_respects_explicit_model_and_base_url() -> None:
    config = resolve_config(
        make_args(
            api_key="cli-key",
            gemini=True,
            model="gemini-2.5-pro",
            base_url="http://custom-gemini/v1",
        )
    )
    assert config.base_url == "http://custom-gemini/v1"
    assert config.model_name == "gemini-2.5-pro"


def test_skills_dir_is_expanded_and_resolved(tmp_path: Path) -> None:
    config = resolve_config(make_args(api_key="k", skills_dir=Path("~/skills")))
    assert config.skills_dir == (tmp_path / "skills").resolve()


def test_openai_env_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai")
    monkeypatch.setenv("OPENAI_MODEL", "openai-model")

    config = resolve_config(make_args())
    assert (config.api_key, config.base_url, config.model_name) == (
        "openai-key",
        "http://openai",
        "openai-model",
    )


def test_cli_args_override_env_and_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_default_config(
        tmp_path,
        "api_key: file-key\nbase_url: http://file\nmodel: file-model\n",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")

    config = resolve_config(
        make_args(api_key="cli-key", base_url="http://cli", model="cli-model")
    )
    assert (config.api_key, config.base_url, config.model_name) == (
        "cli-key",
        "http://cli",
        "cli-model",
    )


def test_env_overrides_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_default_config(
        tmp_path,
        "api_key: file-key\nbase_url: http://file\nmodel: file-model\nmax_retries: 2\n",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("RPI_AGENT_MAX_RETRIES", "7")

    config = resolve_config(make_args())
    assert (
        config.api_key,
        config.base_url,
        config.model_name,
        config.max_retries,
    ) == ("env-key", "http://env", "env-model", 7)


def test_config_file_supplies_all_options(tmp_path: Path) -> None:
    write_default_config(
        tmp_path,
        "api-key: file-key\n"
        "base-url: http://file/v1\n"
        "model_name: custom-model\n"
        "skills-dir: ~/my-skills\n"
        "yolo: true\n"
        "no-prune: true\n"
        "max-retries: 3\n",
    )

    config = resolve_config(make_args())
    assert config.api_key == "file-key"
    assert config.base_url == "http://file/v1"
    assert config.model_name == "custom-model"
    assert config.skills_dir == (tmp_path / "my-skills").resolve()
    assert config.yolo is True
    assert config.prune_context is False
    assert config.max_retries == 3


def test_config_file_gemini_mode_and_prune_context(tmp_path: Path) -> None:
    write_default_config(
        tmp_path,
        "api_key: gemini-key\ngemini: true\nprune_context: false\n",
    )

    config = resolve_config(make_args())
    assert config.api_key == "gemini-key"
    assert config.base_url == GEMINI_BASE_URL
    assert config.model_name == DEFAULT_GEMINI_MODEL
    assert config.prune_context is False


def test_explicit_config_path_loaded(tmp_path: Path) -> None:
    custom = tmp_path / "custom.yml"
    custom.write_text("api_key: custom-key\nmodel: custom-m\n", encoding="utf-8")

    config = resolve_config(make_args(config=custom))
    assert config.api_key == "custom-key"
    assert config.model_name == "custom-m"


def test_explicit_config_path_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        load_user_config(tmp_path / "nonexistent.yml")


def test_empty_config_file_returns_empty_dict(tmp_path: Path) -> None:
    write_default_config(tmp_path, "# just a comment\n")
    assert load_user_config() == {}


def test_config_file_read_or_yaml_error_raises(tmp_path: Path) -> None:
    write_default_config(tmp_path, "api_key: [unterminated\n")
    with pytest.raises(ConfigurationError, match="Failed to load configuration file"):
        load_user_config()


def test_config_file_non_mapping_raises(tmp_path: Path) -> None:
    write_default_config(tmp_path, "- item1\n- item2\n")
    with pytest.raises(ConfigurationError, match="must contain a YAML mapping"):
        load_user_config()


def test_config_file_non_string_key_raises(tmp_path: Path) -> None:
    write_default_config(tmp_path, "123: value\n")
    with pytest.raises(
        ConfigurationError, match="Configuration keys .* must be strings"
    ):
        load_user_config()


def test_config_file_unknown_option_raises(tmp_path: Path) -> None:
    write_default_config(tmp_path, "unknown_option: 1\n")
    with pytest.raises(
        ConfigurationError, match="Unknown configuration option 'unknown_option'"
    ):
        load_user_config()


@pytest.mark.parametrize(
    ("yaml_text", "match"),
    [
        ("api_key: ''\n", "must be a non-empty string"),
        ("api_key: 123\n", "must be a non-empty string"),
        ("gemini: 'yes'\n", "must be true or false"),
        ("max_retries: true\n", "must be an integer"),
        ("max_retries: '5'\n", "must be an integer"),
    ],
)
def test_config_file_invalid_value_types_raise(
    tmp_path: Path, yaml_text: str, match: str
) -> None:
    write_default_config(tmp_path, yaml_text)
    with pytest.raises(ConfigurationError, match=match):
        load_user_config()


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

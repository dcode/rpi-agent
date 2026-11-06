from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from google.adk.events import Event
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types

from rpi_agent.constants import APP_NAME, USER_ID
from rpi_agent.exceptions import ConfigurationError
from rpi_agent.sessions import (
    LATEST_SESSION,
    count_user_turns,
    create_session,
    list_sessions,
    load_session,
    open_session_service,
)


def test_open_session_service_uses_state_dir(tmp_path: Path) -> None:
    service = open_session_service(tmp_path)
    assert isinstance(service, SqliteSessionService)

    async def scenario() -> None:
        await create_session(service)

    asyncio.run(scenario())
    assert (tmp_path / "sessions.db").is_file()


def test_sessions_persist_across_service_instances(tmp_path: Path) -> None:
    async def scenario() -> tuple[str, list[str], str]:
        first = open_session_service(tmp_path)
        older = await create_session(first)
        await asyncio.sleep(0.01)
        newer = await create_session(first)
        await first.close()

        second = open_session_service(tmp_path)
        listed = [s.id for s in await list_sessions(second)]
        latest = (await load_session(second, LATEST_SESSION)).id
        assert (await load_session(second, older)).id == older
        return newer, listed, latest

    newer, listed, latest = asyncio.run(scenario())
    assert listed[0] == newer
    assert len(listed) == 2
    assert latest == newer


def test_create_session_ids_are_unique(tmp_path: Path) -> None:
    async def scenario() -> set[str]:
        service = open_session_service(tmp_path)
        return {await create_session(service) for _ in range(5)}

    ids = asyncio.run(scenario())
    assert len(ids) == 5
    assert all(i.startswith("sess_") for i in ids)


def test_load_latest_without_sessions(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="No saved sessions"):
        asyncio.run(load_session(open_session_service(tmp_path), LATEST_SESSION))


def test_load_unknown_session(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Session 'sess_nope' not found"):
        asyncio.run(load_session(open_session_service(tmp_path), "sess_nope"))


def test_count_user_turns(tmp_path: Path) -> None:
    async def scenario() -> int:
        service = open_session_service(tmp_path)
        session = await service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id="s"
        )
        for author in ("user", "rpi_root_agent", "user", "rpi_root_agent"):
            await service.append_event(
                session,
                Event(
                    author=author,
                    invocation_id="inv",
                    content=types.Content(role="user", parts=[types.Part(text="x")]),
                ),
            )
        return count_user_turns(await load_session(service, "s"))

    assert asyncio.run(scenario()) == 2

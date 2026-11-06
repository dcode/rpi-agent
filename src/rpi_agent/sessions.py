"""Persistent conversation sessions backed by SQLite."""

from __future__ import annotations

import uuid
from pathlib import Path

from google.adk.sessions import BaseSessionService, Session
from google.adk.sessions.sqlite_session_service import SqliteSessionService

from rpi_agent.constants import APP_NAME, USER_ID
from rpi_agent.exceptions import ConfigurationError

LATEST_SESSION: str = "latest"


def open_session_service(state_dir: Path) -> SqliteSessionService:
    """Opens the repository's session database at `<state_dir>/sessions.db`."""
    return SqliteSessionService(str(state_dir / "sessions.db"))


async def create_session(service: BaseSessionService) -> str:
    """Creates a new empty session and returns its ID."""
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    await service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    return session_id


async def list_sessions(service: BaseSessionService) -> list[Session]:
    """Returns stored sessions, most recently updated first (without events)."""
    response = await service.list_sessions(app_name=APP_NAME, user_id=USER_ID)
    return sorted(response.sessions, key=lambda s: s.last_update_time, reverse=True)


async def load_session(service: BaseSessionService, requested: str) -> Session:
    """Loads a session by ID, or the most recent one for `LATEST_SESSION`.

    Raises:
        ConfigurationError: If no matching session exists.
    """
    session_id = requested
    if requested == LATEST_SESSION:
        sessions = await list_sessions(service)
        if not sessions:
            raise ConfigurationError("No saved sessions to resume in this repository.")
        session_id = sessions[0].id

    session = await service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    if session is None:
        raise ConfigurationError(
            f"Session '{requested}' not found. Use /sessions to list saved sessions."
        )
    return session


def count_user_turns(session: Session) -> int:
    """Number of prompts the user sent in a session."""
    return sum(1 for event in session.events if event.author == "user")

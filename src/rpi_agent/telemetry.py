"""Structured logging with structlog, OpenTelemetry tracing, and HTTP telemetry."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import litellm
import structlog
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status, StatusCode, Tracer

from rpi_agent.config import normalize_log_level
from rpi_agent.constants import DEFAULT_LOG_BUFFER_MAX_RECORDS, DEFAULT_LOG_LEVEL

_SENSITIVE_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "api-key",
        "x-api-key",
        "x-goog-api-key",
        "proxy-authorization",
        "cookie",
        "set-cookie",
    }
)

_MAX_BODY_LOG_CHARS: int = 2000


def get_logger(name: str = "rpi_agent") -> structlog.stdlib.BoundLogger:
    """Returns a structlog bound logger for the given module or component name."""
    return structlog.stdlib.get_logger(name)


def redact_headers(headers: Any) -> dict[str, str]:
    """Redacts sensitive authentication headers for safe telemetry logging."""
    if not headers:
        return {}
    redacted: dict[str, str] = {}
    items = headers.items() if hasattr(headers, "items") else headers
    for key, value in items:
        key_str = str(key)
        val_str = str(value)
        if key_str.lower() in _SENSITIVE_HEADERS:
            redacted[key_str] = "***REDACTED***"
        else:
            redacted[key_str] = val_str
    return redacted


def _truncate_text(text: str, max_chars: int = _MAX_BODY_LOG_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def add_otel_context(
    _logger: Any, _method_name: str, event_dict: structlog.typing.EventDict
) -> structlog.typing.EventDict:
    """Structlog processor that injects active OpenTelemetry trace_id and span_id."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        event_dict.setdefault("trace_id", format(ctx.trace_id, "032x"))
        event_dict.setdefault("span_id", format(ctx.span_id, "016x"))
    return event_dict


@dataclass(slots=True)
class LogEntry:
    """A single structured log entry stored in the session log buffer.

    Attributes:
        timestamp: ISO-8601 UTC timestamp.
        level: Uppercase log level name (e.g. 'DEBUG', 'INFO', 'WARNING').
        level_no: Numeric Python logging level (e.g. 10, 20, 30, 40, 50).
        logger_name: Originating logger name.
        event: Primary event message or identifier.
        fields: Structured key-value context fields (including trace_id/span_id).
    """

    timestamp: str
    level: str
    level_no: int
    logger_name: str
    event: str
    fields: dict[str, Any] = field(default_factory=dict)

    def format_line(self) -> str:
        """Formats the log entry as a concise human-readable line for the log pane."""
        short_time = (
            self.timestamp[11:23] if len(self.timestamp) >= 23 else self.timestamp
        )
        pairs: list[str] = []
        for k, v in self.fields.items():
            if isinstance(v, (dict, list)):
                val_repr = json.dumps(v, default=str, ensure_ascii=False)
            else:
                val_repr = str(v)
            pairs.append(f"{k}={val_repr}")
        kv_suffix = f" {' '.join(pairs)}" if pairs else ""
        return f"{short_time} [{self.level:<7}] {self.event}{kv_suffix}"

    def to_json(self) -> str:
        """Serializes the log entry as a single JSON line for disk persistence."""
        payload: dict[str, Any] = {
            "timestamp": self.timestamp,
            "level": self.level,
            "logger": self.logger_name,
            "event": self.event,
            **self.fields,
        }
        return json.dumps(payload, default=str, ensure_ascii=False)


class SessionLogBuffer(logging.Handler):
    """Ring-buffer logging handler supporting live UI subscriptions and disk export."""

    def __init__(
        self,
        max_records: int = DEFAULT_LOG_BUFFER_MAX_RECORDS,
        display_level: str = DEFAULT_LOG_LEVEL,
        log_file: Path | None = None,
    ) -> None:
        """Initializes the session log buffer.

        Args:
            max_records: Maximum number of log entries retained in memory.
            display_level: Minimum log level shown in the REPL log pane.
            log_file: Optional path to stream logs to continuously.
        """
        super().__init__(level=logging.DEBUG)
        self.max_records = max_records
        self._records: deque[LogEntry] = deque(maxlen=max_records)
        self._display_level: str = normalize_log_level(display_level)
        self._display_level_no: int = getattr(logging, self._display_level)
        self._log_file: Path | None = log_file
        self._listeners: list[Callable[[], None]] = []

    @property
    def display_level(self) -> str:
        """Current minimum log level displayed in the log pane and streamed to disk."""
        return self._display_level

    @display_level.setter
    def display_level(self, level: str) -> None:
        normalized = normalize_log_level(level)
        self._display_level = normalized
        self._display_level_no = getattr(logging, normalized)
        self._notify_listeners()

    @property
    def log_file(self) -> Path | None:
        """Optional file path where logs are continuously appended."""
        return self._log_file

    @log_file.setter
    def log_file(self, path: Path | None) -> None:
        self._log_file = path

    @property
    def records(self) -> list[LogEntry]:
        """Returns a snapshot of all buffered log entries."""
        return list(self._records)

    def add_listener(self, callback: Callable[[], None]) -> None:
        """Registers a callback invoked whenever new logs arrive or display level changes."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[], None]) -> None:
        """Unregisters a previously added listener callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self) -> None:
        for listener in list(self._listeners):
            listener()

    def emit(self, record: logging.LogRecord) -> None:
        """Captures a Python or structlog LogRecord into the ring buffer."""
        entry = self._record_to_entry(record)
        self._records.append(entry)

        if self._log_file is not None and entry.level_no >= self._display_level_no:
            self._append_to_file(self._log_file, entry)

        self._notify_listeners()

    def filter_entries(self, min_level: str | None = None) -> list[LogEntry]:
        """Returns buffered entries at or above `min_level` (defaults to `display_level`)."""
        target_level = (
            normalize_log_level(min_level)
            if min_level is not None
            else self._display_level
        )
        target_no: int = getattr(logging, target_level)
        return [e for e in self._records if e.level_no >= target_no]

    def format_for_display(self, min_level: str | None = None) -> str:
        """Formats matching buffered entries as newline-separated lines for the UI pane."""
        entries = self.filter_entries(min_level)
        if not entries:
            lvl = (
                normalize_log_level(min_level)
                if min_level is not None
                else self._display_level
            )
            return f"[No log entries at level {lvl} or higher]"
        return "\n".join(entry.format_line() for entry in entries)

    def save_to_file(self, path: Path, min_level: str | None = None) -> int:
        """Writes buffered log entries at or above `min_level` to `path` as JSONL.

        Args:
            path: Destination file path on disk.
            min_level: Optional minimum log level name (defaults to `display_level`).

        Returns:
            Number of log entries written.
        """
        entries = self.filter_entries(min_level)
        target = path.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [entry.to_json() for entry in entries]
        content = "\n".join(lines) + ("\n" if lines else "")
        target.write_text(content, encoding="utf-8")
        return len(entries)

    def clear(self) -> None:
        """Clears all buffered log entries and notifies listeners."""
        self._records.clear()
        self._notify_listeners()

    @staticmethod
    def _append_to_file(path: Path, entry: LogEntry) -> None:
        try:
            target = path.expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(entry.to_json() + "\n")
        except OSError:
            pass

    @staticmethod
    def _record_to_entry(record: logging.LogRecord) -> LogEntry:
        level_name = record.levelname.upper()
        level_no = record.levelno
        logger_name = record.name

        if isinstance(record.msg, dict):
            raw_dict = dict(record.msg)
            event = str(raw_dict.pop("event", ""))
            timestamp = str(
                raw_dict.pop(
                    "timestamp",
                    datetime.fromtimestamp(record.created, UTC).isoformat(),
                )
            )
            raw_dict.pop("level", None)
            logger_name = str(raw_dict.pop("logger", logger_name))
            fields = raw_dict
        else:
            event = record.getMessage()
            timestamp = datetime.fromtimestamp(record.created, UTC).isoformat()
            fields = {"logger": logger_name} if logger_name else {}
            if record.exc_info and record.exc_info[1] is not None:
                fields["exception"] = str(record.exc_info[1])

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            fields.setdefault("trace_id", format(ctx.trace_id, "032x"))
            fields.setdefault("span_id", format(ctx.span_id, "016x"))

        return LogEntry(
            timestamp=timestamp,
            level=level_name,
            level_no=level_no,
            logger_name=logger_name,
            event=event,
            fields=fields,
        )


class StructlogSpanExporter(SpanExporter):
    """OpenTelemetry SpanExporter that emits completed spans through structlog."""

    def __init__(self, logger_name: str = "rpi_agent.otel") -> None:
        self._logger = get_logger(logger_name)

    def export(self, spans: Any) -> SpanExportResult:
        """Exports a batch of completed OpenTelemetry spans to structlog."""
        for span in spans:
            self._log_span(span)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shuts down the exporter."""

    def _log_span(self, span: ReadableSpan) -> None:
        ctx = span.context
        trace_id = format(ctx.trace_id, "032x") if ctx else ""
        span_id = format(ctx.span_id, "016x") if ctx else ""
        parent_id = format(span.parent.span_id, "016x") if span.parent else None
        duration_ms = (
            round((span.end_time - span.start_time) / 1e6, 2)
            if span.end_time is not None and span.start_time is not None
            else 0.0
        )
        attrs = dict(span.attributes) if span.attributes else {}
        status_code = span.status.status_code.name if span.status else "UNSET"

        payload: dict[str, Any] = {
            "span_name": span.name,
            "trace_id": trace_id,
            "span_id": span_id,
            "duration_ms": duration_ms,
            "status": status_code,
        }
        if parent_id:
            payload["parent_span_id"] = parent_id
        for k, v in attrs.items():
            payload[str(k)] = v

        if span.status and span.status.status_code is StatusCode.ERROR:
            if span.status.description:
                payload["error_description"] = span.status.description
            self._logger.warning("otel.span.error", **payload)
        else:
            self._logger.debug("otel.span.completed", **payload)


# State for idempotent HTTP instrumentation
_ORIGINAL_ASYNC_CLIENT_INIT: Any = None
_ORIGINAL_SYNC_CLIENT_INIT: Any = None
_ACTIVE_TRACER_PROVIDER: TracerProvider | None = None
_ACTIVE_LOG_BUFFER: SessionLogBuffer | None = None


def get_tracer(name: str = "rpi_agent") -> Tracer:
    """Returns an OpenTelemetry Tracer backed by the active TracerProvider."""
    if _ACTIVE_TRACER_PROVIDER is not None:
        return _ACTIVE_TRACER_PROVIDER.get_tracer(name)
    return trace.get_tracer(name)


def _extract_request_body_preview(request: httpx.Request) -> str | None:
    try:
        raw = request.content
    except httpx.RequestNotRead:
        return None
    if not raw:
        return None
    try:
        decoded = raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None
    return _truncate_text(decoded)


def _extract_response_body_preview(response: httpx.Response) -> str | None:
    if not getattr(response, "is_stream_consumed", False) and not hasattr(
        response, "_content"
    ):
        return None
    try:
        raw = response.content
    except httpx.ResponseNotRead:
        return None
    if not raw:
        return None
    try:
        decoded = raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None
    return _truncate_text(decoded)


def _on_http_request_sync(request: httpx.Request) -> None:
    logger = get_logger("rpi_agent.http")
    tracer = get_tracer("rpi_agent.http")
    url_str = str(request.url)
    method = request.method
    span = tracer.start_span(
        f"HTTP {method}",
        attributes={
            "http.request.method": method,
            "url.full": url_str,
            "server.address": request.url.host or "",
        },
    )
    request.extensions["rpi_otel_span"] = span
    request.extensions["rpi_start_time"] = time.perf_counter()

    with trace.use_span(span, end_on_exit=False):
        logger.info(
            "http.request",
            method=method,
            url=url_str,
            host=request.url.host or "",
        )
        body_preview = _extract_request_body_preview(request)
        logger.debug(
            "http.request.detail",
            method=method,
            url=url_str,
            headers=redact_headers(request.headers),
            body=body_preview,
        )


def _on_http_response_sync(response: httpx.Response) -> None:
    logger = get_logger("rpi_agent.http")
    request = response.request
    span = request.extensions.pop("rpi_otel_span", None)
    start_time = request.extensions.pop("rpi_start_time", None)
    duration_ms = (
        round((time.perf_counter() - start_time) * 1000.0, 2)
        if isinstance(start_time, float)
        else 0.0
    )
    status_code = response.status_code
    method = request.method
    url_str = str(request.url)

    if span is not None:
        span.set_attribute("http.response.status_code", status_code)
        span.set_attribute("http.duration_ms", duration_ms)
        if status_code >= 400:
            span.set_status(Status(StatusCode.ERROR, f"HTTP {status_code}"))
        else:
            span.set_status(Status(StatusCode.OK))

    ctx_mgr = (
        trace.use_span(span, end_on_exit=True)
        if span is not None
        else _null_span_context()
    )
    with ctx_mgr:
        log_fn = logger.warning if status_code >= 400 else logger.info
        log_fn(
            "http.response",
            method=method,
            url=url_str,
            status_code=status_code,
            duration_ms=duration_ms,
        )
        body_preview = _extract_response_body_preview(response)
        logger.debug(
            "http.response.detail",
            method=method,
            url=url_str,
            status_code=status_code,
            duration_ms=duration_ms,
            headers=redact_headers(response.headers),
            body=body_preview,
        )


async def _on_http_request_async(request: httpx.Request) -> None:
    _on_http_request_sync(request)


async def _on_http_response_async(response: httpx.Response) -> None:
    _on_http_response_sync(response)


class _null_span_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


def install_httpx_instrumentation() -> None:
    """Installs OpenTelemetry and structlog request/response hooks onto httpx clients."""
    global _ORIGINAL_ASYNC_CLIENT_INIT, _ORIGINAL_SYNC_CLIENT_INIT

    if _ORIGINAL_ASYNC_CLIENT_INIT is None:
        orig_async = httpx.AsyncClient.__init__
        _ORIGINAL_ASYNC_CLIENT_INIT = orig_async

        def _patched_async_init(
            self: httpx.AsyncClient, *args: Any, **kwargs: Any
        ) -> None:
            orig_async(self, *args, **kwargs)
            req_hooks = list(self.event_hooks.get("request", []))
            if _on_http_request_async not in req_hooks:
                req_hooks.append(_on_http_request_async)
            res_hooks = list(self.event_hooks.get("response", []))
            if _on_http_response_async not in res_hooks:
                res_hooks.append(_on_http_response_async)
            self.event_hooks["request"] = req_hooks
            self.event_hooks["response"] = res_hooks

        httpx.AsyncClient.__init__ = _patched_async_init  # type: ignore[method-assign]

    if _ORIGINAL_SYNC_CLIENT_INIT is None:
        orig_sync = httpx.Client.__init__
        _ORIGINAL_SYNC_CLIENT_INIT = orig_sync

        def _patched_sync_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
            orig_sync(self, *args, **kwargs)
            req_hooks = list(self.event_hooks.get("request", []))
            if _on_http_request_sync not in req_hooks:
                req_hooks.append(_on_http_request_sync)
            res_hooks = list(self.event_hooks.get("response", []))
            if _on_http_response_sync not in res_hooks:
                res_hooks.append(_on_http_response_sync)
            self.event_hooks["request"] = req_hooks
            self.event_hooks["response"] = res_hooks

        httpx.Client.__init__ = _patched_sync_init  # type: ignore[method-assign]


def uninstall_httpx_instrumentation() -> None:
    """Restores original httpx.Client and httpx.AsyncClient constructors."""
    global _ORIGINAL_ASYNC_CLIENT_INIT, _ORIGINAL_SYNC_CLIENT_INIT
    if _ORIGINAL_ASYNC_CLIENT_INIT is not None:
        httpx.AsyncClient.__init__ = _ORIGINAL_ASYNC_CLIENT_INIT
        _ORIGINAL_ASYNC_CLIENT_INIT = None
    if _ORIGINAL_SYNC_CLIENT_INIT is not None:
        httpx.Client.__init__ = _ORIGINAL_SYNC_CLIENT_INIT
        _ORIGINAL_SYNC_CLIENT_INIT = None


def _quiet_litellm_console_handlers() -> None:
    """Prevents LiteLLM's internal loggers from printing raw debug lines to stderr."""
    litellm.suppress_debug_info = True
    for name in ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def setup_logging(
    log_level: str = DEFAULT_LOG_LEVEL,
    log_file: Path | None = None,
    max_records: int = DEFAULT_LOG_BUFFER_MAX_RECORDS,
) -> SessionLogBuffer:
    """Configures structlog, stdlib logging, OpenTelemetry tracing, and httpx hooks.

    All log events (DEBUG and above) are captured into the returned `SessionLogBuffer`
    so the user can dynamically inspect or save logs at any verbosity level during
    the session, while the active display and file streaming level defaults to
    `log_level` (WARNING by default, matching prior behavior).

    Args:
        log_level: Initial display and file-streaming log level.
        log_file: Optional file path to stream structured JSONL logs to.
        max_records: Capacity of the in-memory session log ring buffer.

    Returns:
        Configured `SessionLogBuffer` attached to the root logger.
    """
    global _ACTIVE_TRACER_PROVIDER, _ACTIVE_LOG_BUFFER

    normalized_level = normalize_log_level(log_level)
    buffer = SessionLogBuffer(
        max_records=max_records,
        display_level=normalized_level,
        log_file=log_file,
    )

    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        if isinstance(existing, SessionLogBuffer):
            root_logger.removeHandler(existing)
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(buffer)

    _quiet_litellm_console_handlers()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            add_otel_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(StructlogSpanExporter()))
    _ACTIVE_TRACER_PROVIDER = provider
    # Set global tracer provider if not already locked by OpenTelemetry SDK
    trace.set_tracer_provider(provider)

    install_httpx_instrumentation()
    _ACTIVE_LOG_BUFFER = buffer

    logger = get_logger("rpi_agent.telemetry")
    logger.info(
        "telemetry.initialized",
        log_level=normalized_level,
        log_file=str(log_file) if log_file else None,
    )
    return buffer


def get_active_log_buffer() -> SessionLogBuffer | None:
    """Returns the most recently configured `SessionLogBuffer`, if any."""
    return _ACTIVE_LOG_BUFFER

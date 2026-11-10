from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import Status, StatusCode

from rpi_agent import telemetry
from rpi_agent.telemetry import (
    LogEntry,
    SessionLogBuffer,
    StructlogSpanExporter,
    _extract_request_body_preview,
    _extract_response_body_preview,
    _null_span_context,
    _on_http_response_sync,
    _truncate_text,
    get_active_log_buffer,
    get_logger,
    get_tracer,
    install_httpx_instrumentation,
    redact_headers,
    setup_logging,
    uninstall_httpx_instrumentation,
)


def test_redact_headers_handles_empty_mapping_and_sequences() -> None:
    assert redact_headers(None) == {}
    assert redact_headers({}) == {}
    assert redact_headers(
        {
            "Authorization": "Bearer secret",
            "X-API-Key": "secret-key",
            "x-goog-api-key": "g-key",
            "Content-Type": "application/json",
        }
    ) == {
        "Authorization": "***REDACTED***",
        "X-API-Key": "***REDACTED***",
        "x-goog-api-key": "***REDACTED***",
        "Content-Type": "application/json",
    }
    assert redact_headers([("cookie", "abc"), ("Accept", "*/*")]) == {
        "cookie": "***REDACTED***",
        "Accept": "*/*",
    }


def test_truncate_text() -> None:
    assert _truncate_text("short", max_chars=10) == "short"
    assert _truncate_text("0123456789abcdef", max_chars=10) == (
        "0123456789... [truncated 6 chars]"
    )


def test_log_entry_formatting_and_json() -> None:
    entry = LogEntry(
        timestamp="2026-11-06T22:00:01.123456Z",
        level="INFO",
        level_no=logging.INFO,
        logger_name="rpi_agent.test",
        event="http.request",
        fields={"method": "POST", "meta": {"a": 1}, "items": [1, 2]},
    )
    line = entry.format_line()
    assert line.startswith("22:00:01.123 [INFO   ] http.request")
    assert "method=POST" in line
    assert 'meta={"a": 1}' in line
    assert "items=[1, 2]" in line

    short_entry = LogEntry(
        timestamp="12:00:00",
        level="DEBUG",
        level_no=logging.DEBUG,
        logger_name="rpi_agent.test",
        event="ping",
    )
    assert short_entry.format_line() == "12:00:00 [DEBUG  ] ping"

    parsed = json.loads(entry.to_json())
    assert parsed["event"] == "http.request"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "rpi_agent.test"
    assert parsed["method"] == "POST"
    assert parsed["meta"] == {"a": 1}


def test_session_log_buffer_listeners_levels_and_clear() -> None:
    buf = SessionLogBuffer(max_records=5, display_level="WARNING")
    events: list[int] = []

    def listener() -> None:
        events.append(len(buf.records))

    buf.add_listener(listener)
    # Adding the same listener twice is a no-op.
    buf.add_listener(listener)

    assert buf.display_level == "WARNING"
    assert buf.format_for_display() == "[No log entries at level WARNING or higher]"
    assert (
        buf.format_for_display("DEBUG") == "[No log entries at level DEBUG or higher]"
    )

    rec_debug = logging.LogRecord(
        name="app",
        level=logging.DEBUG,
        pathname="",
        lineno=1,
        msg="debug msg",
        args=(),
        exc_info=None,
    )
    rec_warn = logging.LogRecord(
        name="app",
        level=logging.WARNING,
        pathname="",
        lineno=2,
        msg="warn msg",
        args=(),
        exc_info=None,
    )
    buf.emit(rec_debug)
    buf.emit(rec_warn)

    assert len(buf.records) == 2
    assert len(buf.filter_entries()) == 1
    assert len(buf.filter_entries("DEBUG")) == 2
    assert "warn msg" in buf.format_for_display()
    assert "debug msg" not in buf.format_for_display()
    assert "debug msg" in buf.format_for_display("DEBUG")

    buf.display_level = "DEBUG"
    assert buf.display_level == "DEBUG"
    assert "debug msg" in buf.format_for_display()

    buf.clear()
    assert buf.records == []

    buf.remove_listener(listener)
    buf.remove_listener(listener)  # removing again is safe
    count_before = len(events)
    buf.emit(rec_warn)
    assert len(events) == count_before


def test_session_log_buffer_stdlib_record_with_exception_and_empty_name() -> None:
    buf = SessionLogBuffer(display_level="DEBUG")
    try:
        raise ValueError("bad input")
    except ValueError as err:
        exc_info = (type(err), err, err.__traceback__)

    rec = logging.LogRecord(
        name="",
        level=logging.ERROR,
        pathname="",
        lineno=1,
        msg="failed",
        args=(),
        exc_info=exc_info,
    )
    buf.emit(rec)
    (entry,) = buf.records
    assert entry.event == "failed"
    assert entry.fields["exception"] == "bad input"
    assert "logger" not in entry.fields


def test_session_log_buffer_file_streaming_and_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream_file = tmp_path / "logs" / "live.jsonl"
    buf = SessionLogBuffer(display_level="INFO", log_file=stream_file)
    assert buf.log_file == stream_file

    new_stream = tmp_path / "logs" / "live2.jsonl"
    buf.log_file = new_stream
    assert buf.log_file == new_stream

    # DEBUG record is not written to live file when display_level is INFO,
    # but is retained in memory so it can be saved at DEBUG level later.
    buf.emit(logging.LogRecord("app", logging.DEBUG, "", 1, "dbg", (), None))
    buf.emit(logging.LogRecord("app", logging.INFO, "", 2, "inf", (), None))
    assert new_stream.is_file()
    streamed_lines = new_stream.read_text(encoding="utf-8").splitlines()
    assert len(streamed_lines) == 1
    assert json.loads(streamed_lines[0])["event"] == "inf"

    # Saving at DEBUG level exports both records.
    export_debug = tmp_path / "export_debug.jsonl"
    assert buf.save_to_file(export_debug, min_level="DEBUG") == 2
    assert len(export_debug.read_text(encoding="utf-8").splitlines()) == 2

    # Saving at ERROR level writes an empty file and returns 0.
    export_err = tmp_path / "export_err.jsonl"
    assert buf.save_to_file(export_err, min_level="ERROR") == 0
    assert export_err.read_text(encoding="utf-8") == ""

    # OSError during background streaming is swallowed so logging never crashes the app.
    bad_target = tmp_path / "not_a_dir"
    bad_target.write_text("file", encoding="utf-8")
    buf.log_file = bad_target / "nested.jsonl"
    buf.emit(logging.LogRecord("app", logging.INFO, "", 3, "still ok", (), None))


def test_otel_span_exporter_and_context_injection(tmp_path: Path) -> None:
    log_file = tmp_path / "otel.jsonl"
    buf = setup_logging(log_level="DEBUG", log_file=log_file)
    assert get_active_log_buffer() is buf

    # Calling setup_logging again replaces the previous SessionLogBuffer cleanly.
    buf = setup_logging(log_level="DEBUG", log_file=log_file)
    tracer = get_tracer("test.component")
    logger = get_logger("test.component")

    with tracer.start_as_current_span("parent_op", attributes={"custom.key": "val"}):
        with tracer.start_as_current_span("child_op") as child_span:
            logger.info("inside_child", step=1)
            child_span.set_status(Status(StatusCode.ERROR, "something failed"))

    events = [e.event for e in buf.records]
    assert "inside_child" in events
    assert "otel.span.error" in events
    assert "otel.span.completed" in events

    inside = next(e for e in buf.records if e.event == "inside_child")
    assert "trace_id" in inside.fields
    assert "span_id" in inside.fields

    err_span = next(e for e in buf.records if e.event == "otel.span.error")
    assert err_span.level == "WARNING"
    assert err_span.fields["span_name"] == "child_op"
    assert err_span.fields["error_description"] == "something failed"
    assert "parent_span_id" in err_span.fields

    ok_span = next(
        e
        for e in buf.records
        if e.event == "otel.span.completed" and e.fields.get("span_name") == "parent_op"
    )
    assert ok_span.fields["custom.key"] == "val"


def test_span_exporter_direct_and_edge_cases() -> None:
    setup_logging(log_level="DEBUG")
    exporter = StructlogSpanExporter()
    assert exporter.export([]) == SpanExportResult.SUCCESS
    exporter.shutdown()

    # Span with no start/end time, no context, no attributes, and ERROR without description
    fake_span = SimpleNamespace(
        name="bare_span",
        context=None,
        parent=None,
        start_time=None,
        end_time=None,
        attributes=None,
        status=SimpleNamespace(status_code=StatusCode.ERROR, description=None),
    )
    assert exporter.export([fake_span]) == SpanExportResult.SUCCESS


def test_get_tracer_fallback_when_no_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry, "_ACTIVE_TRACER_PROVIDER", None)
    tracer = get_tracer("fallback")
    assert tracer is not None


def test_record_to_entry_injects_otel_for_stdlib_logger() -> None:
    buf = setup_logging(log_level="DEBUG")
    tracer = get_tracer("stdlib.test")
    std_logger = logging.getLogger("stdlib.child")
    with tracer.start_as_current_span("stdlib_span"):
        std_logger.info("hello from stdlib")

    entry = next(e for e in buf.records if e.event == "hello from stdlib")
    assert "trace_id" in entry.fields
    assert "span_id" in entry.fields


def test_httpx_sync_and_async_instrumentation() -> None:
    buf = setup_logging(log_level="DEBUG")
    # Calling install again is idempotent.
    install_httpx_instrumentation()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/error":
            return httpx.Response(429, json={"error": "rate limit"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)

    # Sync client (including pre-populated hooks to verify deduplication)
    with httpx.Client(transport=transport) as sync_client:
        r1 = sync_client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": "Bearer sk-secret"},
            json={"model": "gpt-4o"},
        )
        assert r1.status_code == 200
        r2 = sync_client.get("https://api.openai.com/error")
        assert r2.status_code == 429

    with httpx.Client(
        transport=transport,
        event_hooks={
            "request": [telemetry._on_http_request_sync],
            "response": [telemetry._on_http_response_sync],
        },
    ):
        pass

    # Async client (including pre-populated hooks to verify deduplication)
    async def run_async_client() -> None:
        async with httpx.AsyncClient(transport=transport) as async_client:
            resp = await async_client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": "Bearer sk-secret"},
                content=b"x" * 2500,
            )
            assert resp.status_code == 200
        async with httpx.AsyncClient(
            transport=transport,
            event_hooks={
                "request": [telemetry._on_http_request_async],
                "response": [telemetry._on_http_response_async],
            },
        ):
            pass

    asyncio.run(run_async_client())

    req_entries = [e for e in buf.records if e.event == "http.request"]
    res_entries = [e for e in buf.records if e.event == "http.response"]
    detail_entries = [e for e in buf.records if e.event == "http.request.detail"]

    assert len(req_entries) == 3
    assert len(res_entries) == 3
    assert any(
        e.level == "WARNING" and e.fields["status_code"] == 429 for e in res_entries
    )
    assert any(
        e.fields["headers"].get("authorization") == "***REDACTED***"
        for e in detail_entries
    )

    uninstall_httpx_instrumentation()
    # Calling uninstall again is a safe no-op.
    uninstall_httpx_instrumentation()


def test_http_body_extraction_edge_cases() -> None:
    # Streaming request that hasn't been read yet
    def stream_gen():
        yield b"chunk"

    streaming_req = httpx.Request("POST", "https://example.com", content=stream_gen())
    assert _extract_request_body_preview(streaming_req) is None

    empty_req = httpx.Request("GET", "https://example.com")
    assert _extract_request_body_preview(empty_req) is None

    # Unconsumed streaming response
    streaming_res = httpx.Response(200, content=stream_gen(), request=empty_req)
    assert _extract_response_body_preview(streaming_res) is None

    empty_res = httpx.Response(204, content=b"", request=empty_req)
    assert _extract_response_body_preview(empty_res) is None

    # Response where accessing .content raises ResponseNotRead despite having _content
    class UnreadResponse:
        is_stream_consumed = True

        @property
        def content(self) -> bytes:
            raise httpx.ResponseNotRead()

    assert _extract_response_body_preview(cast(Any, UnreadResponse())) is None

    # Request/Response where decode fails on broken bytes object
    class BrokenBytes:
        def __bool__(self) -> bool:
            return True

        def decode(self, *_a: object, **_k: object) -> str:
            raise RuntimeError("decode boom")

    class BrokenContainer:
        is_stream_consumed = True
        content = BrokenBytes()

    assert _extract_request_body_preview(cast(Any, BrokenContainer())) is None
    assert _extract_response_body_preview(cast(Any, BrokenContainer())) is None

    # Response without prior request hook (no span in extensions)
    setup_logging(log_level="DEBUG")
    bare_res = httpx.Response(200, json={"hello": "world"}, request=empty_req)
    _on_http_response_sync(bare_res)

    with _null_span_context():
        pass

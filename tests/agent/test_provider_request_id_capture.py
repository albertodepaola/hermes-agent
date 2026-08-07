"""Server-side x-request-id capture from streaming + non-streaming responses."""

import logging


def _make_agent():
    import run_agent

    class FakeAgent:
        _current_api_request_id = "turn-1:api:0"

    FakeAgent._capture_request_id = run_agent.AIAgent._capture_request_id
    FakeAgent.get_last_provider_request_id = run_agent.AIAgent.get_last_provider_request_id
    return FakeAgent()


def test_capture_from_stream_response_headers(caplog):
    a = _make_agent()

    class Resp:
        headers = {"x-request-id": "req-stream-1", "x-route": "model-api-rust"}

    with caplog.at_level(logging.INFO, logger="run_agent"):
        a._capture_request_id(Resp())
    assert a.get_last_provider_request_id() == "req-stream-1"
    assert any("req-stream-1" in r.message for r in caplog.records)


def test_capture_from_nonstreaming_completion():
    a = _make_agent()
    from agent.chat_completion_helpers import _capture_nonstreaming_request_id

    class Completion:
        _request_id = "req-nonstream-2"

    out = _capture_nonstreaming_request_id(a, Completion())
    assert out is not None  # must return the completion unchanged
    assert a.get_last_provider_request_id() == "req-nonstream-2"


def test_capture_is_fail_open_on_missing_headers():
    a = _make_agent()
    a._capture_request_id(None)  # no response
    a._capture_request_id(object())  # no .headers
    assert a.get_last_provider_request_id() is None

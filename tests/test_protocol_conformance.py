"""JSON-RPC 2.0 and MCP conformance.

These are the tests worth writing for a hand-rolled protocol implementation,
because protocol conformance is one of the few things in an agent stack that
is *exactly* specified: a malformed request has one correct error code, and
either you return it or you do not. No thresholds, no tolerances, no seeds.

The tests that earn their place are the ones covering the cases a happy-path
implementation gets wrong: notifications, the id/notification distinction,
batch rejection, and the tool-error-versus-protocol-error boundary.
"""

from __future__ import annotations

import io
import json

import pytest

from mcp_server_and_agent.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
)
from mcp_server_and_agent.server import PROTOCOL_VERSION, MCPServer, serve_stdio
from mcp_server_and_agent.tools import SUPPORTED_SCHEMA_KEYWORDS, TOOLS


def msg(method: str, params: dict | None = None, id_: object = 1) -> str:
    payload: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        payload["params"] = params
    if id_ is not _NO_ID:
        payload["id"] = id_
    return json.dumps(payload)


class _NoId:
    pass


_NO_ID = _NoId()


@pytest.fixture
def server() -> MCPServer:
    s = MCPServer()
    s.handle_message(msg("initialize", {"protocolVersion": PROTOCOL_VERSION}))
    s.handle_message(msg("notifications/initialized", {}, id_=_NO_ID))
    return s


# --- error codes -------------------------------------------------------


def test_invalid_json_is_parse_error() -> None:
    response = MCPServer().handle_message("{not json at all")
    assert response["error"]["code"] == PARSE_ERROR
    # The spec requires the id member even when no id could be recovered.
    assert response["id"] is None
    assert "jsonrpc" in response


def test_wrong_jsonrpc_version_is_invalid_request() -> None:
    response = MCPServer().handle_message(json.dumps({"jsonrpc": "1.0", "method": "ping", "id": 1}))
    assert response["error"]["code"] == INVALID_REQUEST


def test_missing_method_is_invalid_request() -> None:
    response = MCPServer().handle_message(json.dumps({"jsonrpc": "2.0", "id": 1}))
    assert response["error"]["code"] == INVALID_REQUEST


def test_non_object_params_is_invalid_params() -> None:
    raw = json.dumps({"jsonrpc": "2.0", "method": "ping", "params": [1, 2], "id": 1})
    assert MCPServer().handle_message(raw)["error"]["code"] == INVALID_PARAMS


def test_batch_is_rejected_rather_than_half_implemented() -> None:
    response = MCPServer().handle_message(json.dumps([{"jsonrpc": "2.0", "method": "ping"}]))
    assert response["error"]["code"] == INVALID_REQUEST
    assert "batch" in response["error"]["data"]


def test_unknown_method_is_method_not_found(server: MCPServer) -> None:
    assert server.handle_message(msg("no/such/method"))["error"]["code"] == METHOD_NOT_FOUND


def test_internal_error_boundary_does_not_kill_the_session(server: MCPServer) -> None:
    """An unexpected exception must become -32603, not escape the loop.

    A handler that raises through would kill the stdio loop and leave the
    client waiting on a response that never arrives -- strictly worse than
    an error object.
    """
    import mcp_server_and_agent.server as mod

    original = mod.MCPServer._tools_list
    try:
        mod.MCPServer._tools_list = lambda self, params: 1 / 0  # noqa: ARG005
        response = server.handle_message(msg("tools/list"))
    finally:
        mod.MCPServer._tools_list = original

    assert response["error"]["code"] == INTERNAL_ERROR
    # Session survives.
    assert "result" in server.handle_message(msg("ping"))


# --- lifecycle ---------------------------------------------------------


def test_requests_before_initialize_are_rejected() -> None:
    response = MCPServer().handle_message(msg("tools/list"))
    assert response["error"]["code"] == INVALID_REQUEST
    assert "initialize" in response["error"]["data"]


def test_initialize_negotiates_rather_than_echoes() -> None:
    """The server replies with the version it speaks, not the one requested."""
    response = MCPServer().handle_message(msg("initialize", {"protocolVersion": "1999-01-01"}))
    assert response["result"]["protocolVersion"] == PROTOCOL_VERSION


def test_initialize_requires_protocol_version() -> None:
    assert MCPServer().handle_message(msg("initialize", {}))["error"]["code"] == INVALID_PARAMS


def test_notifications_get_no_response(server: MCPServer) -> None:
    assert server.handle_message(msg("notifications/initialized", {}, id_=_NO_ID)) is None


def test_unknown_notification_is_silently_ignored(server: MCPServer) -> None:
    """Replying to a notification corrupts a stream that cannot resynchronise."""
    assert server.handle_message(msg("no/such/notification", {}, id_=_NO_ID)) is None


def test_null_id_is_a_request_not_a_notification(server: MCPServer) -> None:
    """`"id": null` is a request with a null id. Absence of `id` is a
    notification. Only membership testing separates them."""
    response = server.handle_message(msg("ping", {}, id_=None))
    assert response is not None
    assert response["id"] is None
    assert response["result"] == {}


# --- tools -------------------------------------------------------------


def test_tools_list_exposes_schemas_and_annotations(server: MCPServer) -> None:
    tools = server.handle_message(msg("tools/list"))["result"]["tools"]
    assert len(tools) >= 3
    for t in tools:
        assert t["inputSchema"]["type"] == "object"
        assert "destructiveHint" in t["annotations"]
    destructive = [t["name"] for t in tools if t["annotations"]["destructiveHint"]]
    assert destructive == ["delete_record"]


def test_declared_schemas_stay_inside_the_validated_subset() -> None:
    """The validator covers a subset of JSON Schema, so the descriptors must
    not use anything outside it. Otherwise an unsupported keyword silently
    validates every input, which is worse than no validation."""
    for tool in TOOLS.values():
        schema = tool.input_schema
        assert set(schema) <= SUPPORTED_SCHEMA_KEYWORDS | {"properties"}
        for spec in schema["properties"].values():
            assert set(spec) <= SUPPORTED_SCHEMA_KEYWORDS


def test_unknown_tool_is_a_protocol_error_not_a_tool_error(server: MCPServer) -> None:
    """The boundary the whole design rests on. A tool that does not exist
    cannot have failed, so this is -32602 and not isError."""
    response = server.handle_message(msg("tools/call", {"name": "nope", "arguments": {}}))
    assert response["error"]["code"] == INVALID_PARAMS
    assert "unknown tool" in response["error"]["data"]


def test_failing_tool_is_a_result_not_a_protocol_error(server: MCPServer) -> None:
    """The other side of that boundary: the call succeeded, the tool failed,
    and the agent needs the text to decide what to do next."""
    response = server.handle_message(
        msg("tools/call", {"name": "fetch_record", "arguments": {"record_id": "rec-999"}})
    )
    assert "error" not in response
    result = response["result"]
    assert result["isError"] is True
    # Errors are model feedback: the message must name the recovery path.
    assert "search_records" in result["content"][0]["text"]


def test_missing_required_argument_is_invalid_params(server: MCPServer) -> None:
    response = server.handle_message(msg("tools/call", {"name": "fetch_record", "arguments": {}}))
    assert response["error"]["code"] == INVALID_PARAMS


def test_wrong_argument_type_is_invalid_params(server: MCPServer) -> None:
    response = server.handle_message(
        msg("tools/call", {"name": "fetch_record", "arguments": {"record_id": 5}})
    )
    assert response["error"]["code"] == INVALID_PARAMS


def test_bool_is_not_accepted_where_an_integer_is_declared() -> None:
    """bool subclasses int in Python. A validator that forgets this passes
    True for an integer field and fails inside the tool instead."""
    from mcp_server_and_agent.protocol import JsonRpcError
    from mcp_server_and_agent.tools import validate_against_schema

    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    validate_against_schema(schema, {"n": 3}, "t")
    with pytest.raises(JsonRpcError):
        validate_against_schema(schema, {"n": True}, "t")


def test_unknown_argument_is_rejected(server: MCPServer) -> None:
    response = server.handle_message(
        msg("tools/call", {"name": "fetch_record", "arguments": {"record_id": "rec-001", "x": 1}})
    )
    assert response["error"]["code"] == INVALID_PARAMS


# --- the confirmation gate ---------------------------------------------


def test_delete_requires_a_token_the_caller_must_first_be_given(server: MCPServer) -> None:
    """A bare confirm flag is not a gate: a model that guesses it walks
    through. Requiring a token issued by a prior call costs one turn."""
    first = server.handle_message(
        msg("tools/call", {"name": "delete_record", "arguments": {"record_id": "rec-005"}})
    )["result"]
    assert first["isError"] is True
    text = first["content"][0]["text"]
    assert "confirmation required" in text
    # The gate must describe the consequence, not just refuse.
    assert "Access review" in text
    assert server.ctx.records.get("rec-005") is not None

    token = server.ctx.confirm_token("rec-005")
    second = server.handle_message(
        msg(
            "tools/call",
            {
                "name": "delete_record",
                "arguments": {"record_id": "rec-005", "confirm_token": token},
            },
        )
    )["result"]
    assert second["isError"] is False
    assert "rec-005" not in server.ctx.records


def test_a_guessed_token_does_not_open_the_gate(server: MCPServer) -> None:
    response = server.handle_message(
        msg(
            "tools/call",
            {
                "name": "delete_record",
                "arguments": {"record_id": "rec-005", "confirm_token": "yes"},
            },
        )
    )["result"]
    assert response["isError"] is True
    assert "rec-005" in server.ctx.records


# --- idempotency -------------------------------------------------------


def test_replayed_non_idempotent_call_does_not_double_write(server: MCPServer) -> None:
    """Retry-with-backoff is the correct response to a timeout, and without
    dedupe it silently double-writes."""
    call = {
        "name": "append_note",
        "arguments": {"text": "hello"},
        "_meta": {"idempotencyKey": "k-1"},
    }
    server.handle_message(msg("tools/call", call))
    server.handle_message(msg("tools/call", call))
    assert server.ctx.notes == ["hello"]


def test_different_keys_are_distinct_writes(server: MCPServer) -> None:
    for key in ("k-1", "k-2"):
        server.handle_message(
            msg(
                "tools/call",
                {
                    "name": "append_note",
                    "arguments": {"text": "n"},
                    "_meta": {"idempotencyKey": key},
                },
            )
        )
    assert len(server.ctx.notes) == 2


# --- resources and prompts ---------------------------------------------


def test_resource_round_trip(server: MCPServer) -> None:
    listed = server.handle_message(msg("resources/list"))["result"]["resources"]
    uri = listed[0]["uri"]
    body = server.handle_message(msg("resources/read", {"uri": uri}))["result"]
    assert body["contents"][0]["text"].startswith("# Record store schema")


def test_unknown_resource_uri_is_invalid_params(server: MCPServer) -> None:
    response = server.handle_message(msg("resources/read", {"uri": "records://nope"}))
    assert response["error"]["code"] == INVALID_PARAMS


def test_prompt_round_trip(server: MCPServer) -> None:
    listed = server.handle_message(msg("prompts/list"))["result"]["prompts"]
    assert listed[0]["name"] == "audit_owner"
    got = server.handle_message(
        msg("prompts/get", {"name": "audit_owner", "arguments": {"owner": "ops"}})
    )["result"]
    assert "ops" in got["messages"][0]["content"]["text"]


def test_prompt_missing_required_argument_is_invalid_params(server: MCPServer) -> None:
    response = server.handle_message(msg("prompts/get", {"name": "audit_owner", "arguments": {}}))
    assert response["error"]["code"] == INVALID_PARAMS


# --- stdio framing -----------------------------------------------------


def test_stdio_loop_emits_one_line_per_request_and_none_for_notifications() -> None:
    """Framing is newline-delimited JSON. A pretty-printed response would
    split one message across frames, and a reply to a notification would
    desynchronise a client that is not reading one."""
    lines = [
        msg("initialize", {"protocolVersion": PROTOCOL_VERSION}),
        msg("notifications/initialized", {}, id_=_NO_ID),
        msg("tools/list", id_=2),
    ]
    out = io.StringIO()
    serve_stdio(io.StringIO("\n".join(lines) + "\n"), out)

    emitted = [line for line in out.getvalue().split("\n") if line]
    assert len(emitted) == 2  # the notification produced nothing
    for line in emitted:
        json.loads(line)  # each frame is independently valid JSON
    assert json.loads(emitted[1])["id"] == 2


def test_blank_lines_are_ignored() -> None:
    out = io.StringIO()
    serve_stdio(io.StringIO("\n\n" + msg("initialize", {"protocolVersion": "x"}) + "\n"), out)
    assert len([line for line in out.getvalue().split("\n") if line]) == 1

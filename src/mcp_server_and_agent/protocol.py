"""JSON-RPC 2.0 framing and error objects, written against the spec.

This module is deliberately separate from :mod:`server`. JSON-RPC is a
transport-level concern -- it knows about ``id``, ``method``, ``params`` and
error codes, and it knows nothing about tools, resources or prompts. MCP is
the application layer that sits on top. Conflating the two is how you end up
returning ``-32602 Invalid params`` for a tool that failed at runtime, which
tells a client to fix its request when the request was fine.

The distinction that matters most, and the one implementations get wrong:

**A protocol error is not a tool error.** If ``tools/call`` names a tool that
does not exist, that is ``-32602``: the client sent params the server cannot
act on. If the tool exists and *runs* and *fails*, that is a successful
JSON-RPC response whose result carries ``isError: true``. The second case is
model feedback -- the agent reads the error text and tries something else. The
first case is a bug in the client. Collapsing them into one channel means the
agent either retries unfixable calls forever or gives up on recoverable ones.

**Notifications get no response, ever.** A request with no ``id`` member is a
notification. Replying to one -- even with an error -- corrupts the stream for
a client that is not expecting a message, and stdio has no framing to recover
from that. ``handle_message`` returns ``None`` for notifications, including
notifications that are malformed.

Error codes implemented, from the JSON-RPC 2.0 specification:

===========  =====================================================
``-32700``   Parse error -- invalid JSON was received.
``-32600``   Invalid Request -- the JSON is not a valid Request.
``-32601``   Method not found.
``-32602``   Invalid params.
``-32603``   Internal error.
===========  =====================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

JSONRPC_VERSION = "2.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

ERROR_MESSAGES = {
    PARSE_ERROR: "Parse error",
    INVALID_REQUEST: "Invalid Request",
    METHOD_NOT_FOUND: "Method not found",
    INVALID_PARAMS: "Invalid params",
    INTERNAL_ERROR: "Internal error",
}


class JsonRpcError(Exception):
    """A protocol-level failure that must become a JSON-RPC error object.

    Raised by handlers for the cases the *client* has to fix: unknown method,
    unknown tool, params that fail schema validation. A tool that runs and
    fails must NOT raise this -- see the module docstring.
    """

    def __init__(self, code: int, message: str | None = None, data: Any = None) -> None:
        self.code = code
        self.message = message or ERROR_MESSAGES.get(code, "Server error")
        self.data = data
        super().__init__(f"[{code}] {self.message}")

    def to_object(self) -> dict[str, Any]:
        obj: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            obj["data"] = self.data
        return obj


@dataclass(frozen=True)
class Request:
    """A parsed, structurally valid JSON-RPC request or notification."""

    method: str
    params: dict[str, Any]
    id: Any = None
    is_notification: bool = False


def error_response(id_: Any, code: int, message: str | None = None,
                   data: Any = None) -> dict[str, Any]:
    """Build an error response envelope.

    ``id`` is ``null`` when the request could not be parsed far enough to
    recover one -- the spec requires the member to be present regardless, so
    a client correlating by id can still detect the failure rather than
    hanging on a response that never comes.
    """
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": id_,
        "error": JsonRpcError(code, message, data).to_object(),
    }


def result_response(id_: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": id_, "result": result}


def parse_request(raw: str) -> Request:
    """Parse one line of stdio into a Request.

    Raises :class:`JsonRpcError` with ``-32700`` for bad JSON and ``-32600``
    for JSON that is well formed but not a valid Request object.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise JsonRpcError(PARSE_ERROR, data=str(exc)) from exc
    return validate_request(payload)


def validate_request(payload: Any) -> Request:
    """Structural validation of a decoded JSON-RPC request object.

    Batches are rejected with ``-32600`` rather than silently mishandled. MCP
    does not require batching over stdio, and a half-implemented batch path
    that drops responses is worse than an honest refusal.
    """
    if isinstance(payload, list):
        raise JsonRpcError(INVALID_REQUEST, data="batch requests are not supported")
    if not isinstance(payload, dict):
        raise JsonRpcError(INVALID_REQUEST, data="request must be a JSON object")

    if payload.get("jsonrpc") != JSONRPC_VERSION:
        raise JsonRpcError(
            INVALID_REQUEST,
            data=f"jsonrpc must be exactly {JSONRPC_VERSION!r}",
        )

    method = payload.get("method")
    if not isinstance(method, str) or not method:
        raise JsonRpcError(INVALID_REQUEST, data="method must be a non-empty string")

    params = payload.get("params", {})
    # The spec allows params to be an array. MCP only ever uses by-name
    # params, so an array here is a client that has confused protocols.
    if not isinstance(params, dict):
        raise JsonRpcError(INVALID_PARAMS, data="params must be an object")

    # Absence of `id` means notification. `"id": null` is a request with a
    # null id -- legal, and distinct. `in` is the only check that separates
    # them; `.get("id")` cannot.
    is_notification = "id" not in payload
    id_ = payload.get("id")
    if not is_notification and not isinstance(id_, (str, int, type(None))):
        raise JsonRpcError(INVALID_REQUEST, data="id must be a string, number, or null")

    return Request(method=method, params=params, id=id_, is_notification=is_notification)

"""The MCP server: lifecycle, dispatch, and the stdio loop.

Hand-written against the Model Context Protocol specification rather than
built on an SDK. The reason is not purity -- it is that this repo's subject is
what goes wrong between an agent and its tools, and an SDK hides exactly the
seams where that happens: initialization ordering, whether a tool failure is a
protocol error, what a malformed request returns. You cannot measure a layer
you did not write.

Three lifecycle rules the spec imposes, each enforced here and each covered by
a conformance test:

**``initialize`` comes first.** Any other request before it is rejected. A
server that answers ``tools/list`` before negotiating a protocol version has
no idea which version's tool schema it is speaking.

**Version negotiation is a negotiation.** The server replies with a version it
supports. If the client asked for something else, the client decides whether
to continue -- the server does not guess and does not fail the connection.

**``initialize`` is a request; ``notifications/initialized`` is a
notification.** The second gets no response. Sending one back is a stream
corruption that stdio cannot resynchronise from.

The transport is newline-delimited JSON over stdio: one JSON object per line.
That is the framing MCP's stdio transport specifies, and it is why no message
may contain a raw newline.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TextIO

from .protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    JsonRpcError,
    Request,
    error_response,
    parse_request,
    result_response,
)
from .tools import (
    PROMPTS,
    RESOURCE_BODY,
    RESOURCE_URI,
    RESOURCES,
    TOOLS,
    ToolContext,
    ToolResult,
    render_prompt,
    validate_against_schema,
)

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "mcp-server-and-agent", "version": "0.1.0"}


@dataclass
class MCPServer:
    """One MCP session. Stateful by design -- the session owns the store.

    ``initialized`` is not a formality. It is the flag that makes the
    ordering rule testable, and the ordering rule is the one clients break
    most often when they reconnect without redoing the handshake.
    """

    ctx: ToolContext = field(default_factory=ToolContext)
    initialized: bool = False
    # Results of prior non-idempotent calls, keyed by the client's
    # idempotency key. Without this, the correct response to a timeout --
    # retry with backoff -- double-writes. With it, a replayed call returns
    # the original result and touches nothing.
    _idempotency_cache: dict[str, ToolResult] = field(default_factory=dict)
    call_count: int = 0

    # --- dispatch ------------------------------------------------------

    def handle_message(self, raw: str) -> dict[str, Any] | None:
        """Handle one line of input. Returns the response, or None.

        None means "write nothing": the input was a notification, or was
        blank. Every other path returns exactly one envelope.
        """
        if not raw.strip():
            return None

        try:
            request = parse_request(raw)
        except JsonRpcError as exc:
            # A malformed message might have been a notification, but we
            # cannot know that without parsing it -- so we answer with
            # id=null, which the spec prescribes for exactly this case.
            return error_response(None, exc.code, exc.message, exc.data)

        return self.handle_request(request)

    def handle_request(self, request: Request) -> dict[str, Any] | None:
        handler = self._handlers().get(request.method)

        if request.is_notification:
            # Notifications are fire-and-forget in both directions,
            # including when they name a method we do not implement.
            if handler is not None:
                try:
                    handler(request.params)
                except JsonRpcError:
                    pass
            return None

        if handler is None:
            return error_response(
                request.id, METHOD_NOT_FOUND, data=f"unknown method {request.method!r}"
            )

        if not self.initialized and request.method != "initialize":
            return error_response(
                request.id,
                INVALID_REQUEST,
                data=f"{request.method!r} received before 'initialize'",
            )

        try:
            return result_response(request.id, handler(request.params))
        except JsonRpcError as exc:
            return error_response(request.id, exc.code, exc.message, exc.data)
        except Exception as exc:  # noqa: BLE001 - the -32603 boundary
            # An unexpected exception is a server bug, and -32603 is what the
            # spec has for it. Letting it escape would kill the stdio loop and
            # leave the client waiting forever on a response that is never
            # coming, which is a strictly worse failure than an error object.
            return error_response(request.id, INTERNAL_ERROR, data=repr(exc))

    def _handlers(self) -> dict[str, Callable[[dict[str, Any]], Any]]:
        return {
            "initialize": self._initialize,
            "notifications/initialized": self._initialized_notification,
            "ping": lambda _params: {},
            "tools/list": self._tools_list,
            "tools/call": self._tools_call,
            "resources/list": lambda _params: {"resources": RESOURCES},
            "resources/read": self._resources_read,
            "prompts/list": lambda _params: {"prompts": PROMPTS},
            "prompts/get": self._prompts_get,
        }

    # --- lifecycle -----------------------------------------------------

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        if not isinstance(requested, str):
            raise JsonRpcError(INVALID_PARAMS, data="initialize requires 'protocolVersion'")
        self.initialized = True
        return {
            # Reply with what we speak, not what was asked for. The client
            # compares and decides; a server that echoes the client's version
            # back has negotiated nothing.
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": SERVER_INFO,
        }

    def _initialized_notification(self, _params: dict[str, Any]) -> None:
        return None

    # --- tools ---------------------------------------------------------

    def _tools_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"tools": [t.to_descriptor() for t in TOOLS.values()]}

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise JsonRpcError(INVALID_PARAMS, data="tools/call requires a string 'name'")

        tool = TOOLS.get(name)
        if tool is None:
            # -32602, not a tool error: the tool does not exist, so there is
            # no tool that could have failed. See protocol.py's docstring.
            raise JsonRpcError(
                INVALID_PARAMS,
                data=f"unknown tool {name!r}; available: {sorted(TOOLS)}",
            )

        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise JsonRpcError(INVALID_PARAMS, data="'arguments' must be an object")

        validate_against_schema(tool.input_schema, arguments, name)

        key = params.get("_meta", {}).get("idempotencyKey") if isinstance(
            params.get("_meta"), dict
        ) else None
        if key is not None and not tool.idempotent:
            cached = self._idempotency_cache.get(key)
            if cached is not None:
                return cached.to_content()

        self.call_count += 1
        result = tool.handler(self.ctx, arguments)

        if key is not None and not tool.idempotent and not result.is_error:
            # Only successful calls are cached. Replaying a failure should
            # re-run it: the failure may have been transient, and caching it
            # turns one bad moment into a permanent one.
            self._idempotency_cache[key] = result

        return result.to_content()

    # --- resources and prompts -----------------------------------------

    def _resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri")
        if uri != RESOURCE_URI:
            raise JsonRpcError(
                INVALID_PARAMS,
                data=f"unknown resource uri {uri!r}; available: [{RESOURCE_URI!r}]",
            )
        return {
            "contents": [
                {"uri": RESOURCE_URI, "mimeType": "text/markdown", "text": RESOURCE_BODY}
            ]
        }

    def _prompts_get(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise JsonRpcError(INVALID_PARAMS, data="prompts/get requires a string 'name'")
        return render_prompt(name, params.get("arguments", {}) or {})


def serve_stdio(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    """Run the newline-delimited-JSON loop until stdin closes.

    Injectable streams so the loop itself is testable end to end rather than
    only via ``handle_message``. Framing bugs -- a stray newline, a missing
    flush -- only show up at this level.
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = MCPServer()

    for line in stdin:
        response = server.handle_message(line)
        if response is None:
            continue
        # `separators` guarantees no embedded newline from pretty-printing,
        # which would split one message across two frames.
        stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        stdout.flush()


if __name__ == "__main__":  # pragma: no cover - entry point
    serve_stdio()

"""The tool surface: five tools, one resource, one prompt.

Every design decision here exists to be measured by the topology experiment,
so each is stated rather than assumed.

**Schema clarity is an experimental variable, not polish.** Two tools in this
set are deliberately confusable -- ``search_records`` and ``fetch_record``.
Their descriptions overlap on the word "record", and the fault model in
:mod:`faults` can nudge the agent between them. Tool misselection is not an
exotic failure; it is what happens when a tool list grows past the point where
descriptions discriminate. Making it reproducible required building a tool
list where confusion is *possible*.

**Errors are model feedback.** A tool that fails returns ``isError: True`` and
a message written for a reader who must decide what to do next: what was
wrong, and what would work instead. ``fetch_record`` on a missing id does not
say "KeyError"; it says the id is unknown and names how to find valid ones.
The difference is measurable -- an agent can recover from the second.

**Idempotency is declared, and the server enforces it.** ``append_note`` is
not idempotent, so a retry after a timeout double-writes. Each tool carries an
``idempotent`` flag; :mod:`server` uses it to dedupe replayed calls by
``idempotency_key``. Without that, retry-with-jitter -- the correct answer to
a transient failure -- silently corrupts state.

**The destructive tool is gated.** ``delete_record`` refuses unless called
with ``confirm=True`` AND a token issued by a prior unconfirmed call. A single
boolean flag is not a gate: a model that hallucinates ``confirm=True`` on the
first try walks straight through it. Requiring a token the model can only get
by first being told what it is about to delete makes the gate cost one turn,
which is the point.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .protocol import INVALID_PARAMS, JsonRpcError


@dataclass(frozen=True)
class ToolResult:
    """What a tool returns. Mirrors the MCP ``CallToolResult`` shape.

    ``is_error`` being part of the *result* rather than a JSON-RPC error is
    the whole point: the call succeeded at the protocol level and failed at
    the application level, and only the second is something the agent can
    reason about.
    """

    text: str
    is_error: bool = False

    def to_content(self) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": self.text}],
            "isError": self.is_error,
        }


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[ToolContext, dict[str, Any]], ToolResult]
    idempotent: bool = True
    destructive: bool = False

    def to_descriptor(self) -> dict[str, Any]:
        """The ``tools/list`` entry. Annotations follow the MCP tool hints."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {
                "idempotentHint": self.idempotent,
                "destructiveHint": self.destructive,
            },
        }


# --- the backing store -------------------------------------------------

# A fixed corpus, not random data. The topology experiment compares failure
# rates across runs, so the *task* must be identical every time or the
# comparison measures the corpus rather than the topology.
SEED_RECORDS: dict[str, dict[str, Any]] = {
    "rec-001": {
        "title": "Ledger reconciliation", "owner": "ops", "amount": 1420,
        "tags": ["finance"],
    },
    "rec-002": {
        "title": "Quarterly forecast", "owner": "finance", "amount": 9800,
        "tags": ["finance"],
    },
    "rec-003": {
        "title": "Incident postmortem", "owner": "sre", "amount": 0,
        "tags": ["ops", "incident"],
    },
    "rec-004": {
        "title": "Vendor renewal", "owner": "ops", "amount": 3150,
        "tags": ["finance", "vendor"],
    },
    "rec-005": {
        "title": "Access review", "owner": "sre", "amount": 0,
        "tags": ["ops", "security"],
    },
}


@dataclass
class ToolContext:
    """Mutable server state a tool call can read and write.

    Held on the server rather than in a global so a test can construct an
    isolated one, and so the rollback journal is scoped to a single session.
    """

    records: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {k: dict(v) for k, v in SEED_RECORDS.items()}
    )
    notes: list[str] = field(default_factory=list)
    # Tokens handed out by an unconfirmed delete_record call, per record id.
    pending_deletes: dict[str, str] = field(default_factory=dict)
    # Append-only log of state changes, enough to undo them. Compensation,
    # not transactions: stdio has no two-phase commit and pretending
    # otherwise would be the lie this repo is meant to avoid.
    journal: list[tuple[str, Any]] = field(default_factory=list)

    def confirm_token(self, record_id: str) -> str:
        """Deterministic per-record token.

        Deterministic rather than random so a failure-mode reproduction with
        a fixed seed replays identically. The token is a gate against
        accidental deletion, not against an adversary.
        """
        return hashlib.sha256(f"delete:{record_id}".encode()).hexdigest()[:12]

    def rollback(self) -> int:
        """Undo every journalled change, newest first. Returns count undone.

        This is what makes the partial-failure mode *recoverable* rather than
        merely detectable. A multi-step task that dies at step 3 of 5 has left
        two writes behind; without compensation the retry starts from a state
        neither the agent nor the operator can describe.
        """
        undone = 0
        for kind, payload in reversed(self.journal):
            if kind == "delete":
                rid, record = payload
                self.records[rid] = record
            elif kind == "update":
                rid, previous = payload
                if previous is None:
                    self.records.pop(rid, None)
                else:
                    self.records[rid] = previous
            elif kind == "note":
                if self.notes:
                    self.notes.pop()
            undone += 1
        self.journal.clear()
        return undone


# --- schema validation -------------------------------------------------


def validate_against_schema(schema: dict[str, Any], args: dict[str, Any], tool: str) -> None:
    """A deliberately small JSON Schema subset: type, required, enum.

    Not a full validator, and it says so. The alternative was a dependency
    for a repo whose subject is agent topologies, not schema languages. What
    it does cover is what the tool descriptors actually declare, and the
    conformance tests assert that every descriptor stays inside the subset --
    so an unsupported keyword fails the build rather than silently passing
    every input.
    """
    required = schema.get("required", [])
    props = schema.get("properties", {})

    for key in required:
        if key not in args:
            raise JsonRpcError(
                INVALID_PARAMS,
                data=f"tool {tool!r} requires argument {key!r}",
            )

    for key, value in args.items():
        if key not in props:
            raise JsonRpcError(
                INVALID_PARAMS,
                data=f"tool {tool!r} has no argument {key!r}; expected one of {sorted(props)}",
            )
        spec = props[key]
        expected = spec.get("type")
        if expected and not _type_ok(expected, value):
            raise JsonRpcError(
                INVALID_PARAMS,
                data=f"argument {key!r} of {tool!r} must be {expected}, got {type(value).__name__}",
            )
        if "enum" in spec and value not in spec["enum"]:
            raise JsonRpcError(
                INVALID_PARAMS,
                data=f"argument {key!r} of {tool!r} must be one of {spec['enum']}",
            )


_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}

SUPPORTED_SCHEMA_KEYWORDS = frozenset({"type", "properties", "required", "description", "enum"})


def _type_ok(expected: str, value: Any) -> bool:
    types = _TYPE_MAP.get(expected)
    if types is None:
        return True
    # bool is a subclass of int in Python; an integer field must not accept
    # `True`. This is the single most common source of a schema that passes
    # validation and then fails inside the tool.
    if expected in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, types)


# --- tool implementations ----------------------------------------------


def _search_records(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    query = args["query"].lower()
    hits = [
        rid
        for rid, rec in sorted(ctx.records.items())
        if query in rec["title"].lower()
        or query in rec["owner"].lower()
        or any(query in t for t in rec["tags"])
    ]
    if not hits:
        return ToolResult(
            f"no records match {args['query']!r}. "
            f"Known owners: ops, finance, sre. Known tags: finance, ops, incident, "
            f"vendor, security.",
            is_error=True,
        )
    return ToolResult("matched record ids: " + ", ".join(hits))


def _fetch_record(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    rid = args["record_id"]
    rec = ctx.records.get(rid)
    if rec is None:
        # Names the recovery path. An agent that reads "unknown id" and
        # nothing else has no move except to guess again.
        return ToolResult(
            f"unknown record_id {rid!r}. Use search_records to obtain valid ids; "
            f"currently {len(ctx.records)} records exist.",
            is_error=True,
        )
    return ToolResult(
        f"{rid}: title={rec['title']!r} owner={rec['owner']} amount={rec['amount']} "
        f"tags={','.join(rec['tags'])}"
    )


def _summarise_amounts(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    ids = args["record_ids"]
    missing = [r for r in ids if r not in ctx.records]
    if missing:
        return ToolResult(
            f"cannot summarise: unknown ids {missing}. Drop them or resolve them "
            f"with search_records first.",
            is_error=True,
        )
    if not ids:
        return ToolResult("cannot summarise an empty id list.", is_error=True)
    total = sum(ctx.records[r]["amount"] for r in ids)
    return ToolResult(f"count={len(ids)} total={total} mean={total / len(ids):.1f}")


def _append_note(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    ctx.notes.append(args["text"])
    ctx.journal.append(("note", None))
    return ToolResult(f"note appended; {len(ctx.notes)} notes now stored")


def _delete_record(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    rid = args["record_id"]
    if rid not in ctx.records:
        return ToolResult(
            f"unknown record_id {rid!r}; nothing deleted. Use search_records for valid ids.",
            is_error=True,
        )

    token = ctx.confirm_token(rid)
    supplied = args.get("confirm_token")
    if supplied != token:
        ctx.pending_deletes[rid] = token
        rec = ctx.records[rid]
        # The gate's real work: state what is about to be destroyed. A
        # confirmation prompt that does not describe the consequence trains
        # the caller to confirm reflexively.
        return ToolResult(
            f"confirmation required. delete_record would permanently remove {rid} "
            f"(title={rec['title']!r}, owner={rec['owner']}). To proceed, call again "
            f"with confirm_token={token!r}.",
            is_error=True,
        )

    record = ctx.records.pop(rid)
    ctx.pending_deletes.pop(rid, None)
    ctx.journal.append(("delete", (rid, record)))
    return ToolResult(f"deleted {rid}; {len(ctx.records)} records remain")


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            name="search_records",
            # Overlaps with fetch_record on purpose -- see the module
            # docstring. Both descriptions can plausibly answer "get me the
            # record about X", which is what makes misselection reproducible.
            description=(
                "Search the record store by a free-text query matched against title, "
                "owner and tags. Returns matching record ids."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "free-text search term"},
                },
                "required": ["query"],
            },
            handler=_search_records,
        ),
        Tool(
            name="fetch_record",
            description=(
                "Fetch one record by its exact id. Returns the record's title, owner, "
                "amount and tags."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "description": "exact record id, e.g. rec-001"},
                },
                "required": ["record_id"],
            },
            handler=_fetch_record,
        ),
        Tool(
            name="summarise_amounts",
            description=(
                "Compute count, total and mean of the amount field across the given "
                "record ids. All ids must exist."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "record_ids": {"type": "array", "description": "list of exact record ids"},
                },
                "required": ["record_ids"],
            },
            handler=_summarise_amounts,
        ),
        Tool(
            name="append_note",
            description="Append a free-text note to the session note log.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "note body"},
                },
                "required": ["text"],
            },
            handler=_append_note,
            # Two identical appends are two notes. Declared, so the server's
            # dedupe layer knows a retry here is not free.
            idempotent=False,
        ),
        Tool(
            name="delete_record",
            description=(
                "Permanently delete a record. Destructive and gated: the first call "
                "returns a confirm_token describing what would be deleted, and the "
                "deletion only happens on a second call carrying that token."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "description": "exact record id to delete"},
                    "confirm_token": {
                        "type": "string",
                        "description": "token from the prior unconfirmed call",
                    },
                },
                "required": ["record_id"],
            },
            handler=_delete_record,
            idempotent=False,
            destructive=True,
        ),
    ]
}


# --- resources and prompts ---------------------------------------------

RESOURCE_URI = "records://schema"

RESOURCES = [
    {
        "uri": RESOURCE_URI,
        "name": "Record store schema",
        "description": "Field names, types and valid enum values for the record store.",
        "mimeType": "text/markdown",
    }
]

RESOURCE_BODY = """# Record store schema

| field | type | notes |
|---|---|---|
| `id` | string | form `rec-NNN` |
| `title` | string | free text |
| `owner` | string | one of `ops`, `finance`, `sre` |
| `amount` | integer | whole currency units, may be 0 |
| `tags` | array of string | any of `finance`, `ops`, `incident`, `vendor`, `security` |

Records are deleted through a confirmation gate; see `delete_record`.
"""

PROMPTS = [
    {
        "name": "audit_owner",
        "description": "Draft the tool plan for auditing every record belonging to one owner.",
        "arguments": [
            {"name": "owner", "description": "owner to audit", "required": True},
        ],
    }
]


def render_prompt(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name != "audit_owner":
        raise JsonRpcError(INVALID_PARAMS, data=f"unknown prompt {name!r}")
    owner = args.get("owner")
    if not owner:
        raise JsonRpcError(INVALID_PARAMS, data="prompt 'audit_owner' requires argument 'owner'")
    return {
        "description": f"Audit plan for owner {owner}",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"Audit every record owned by {owner}. Use search_records with "
                        f"query={owner!r} to get ids, fetch_record for each id, then "
                        f"summarise_amounts over the full id list. Do not delete anything."
                    ),
                },
            }
        ],
    }

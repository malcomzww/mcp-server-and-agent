"""The scripted agent brain, the ReAct loop, and the task set.

**This brain is simulated.** It contains no model call. It is a deterministic
policy that reads the tool list from the MCP server and chooses the next tool
from a per-task plan, with a fault process (:mod:`faults`) perturbing those
choices at stated rates. Read the module docstring of :mod:`faults` for why
that is the right instrument for this question rather than a compromise.

What the policy is: a plan-and-execute agent with a ReAct-shaped inner loop.
It holds an ordered plan of ``(tool, args)`` steps, executes the head of the
plan, observes, and revises. It is *competent* -- when an error observation
tells it what to do instead, it usually does that (``recovery_skill``). What
makes it fail is the same thing that makes real agents fail: it does not
always parse the error, it sometimes reaches for a neighbouring tool, and it
has a finite step budget and a finite context.

Three controls are separated on purpose, because conflating them is how a
runaway agent gets built:

``max_steps``
    A hard cap. Without one, ``stall`` produces a genuinely infinite loop.
``context_limit``
    A token ceiling on accumulated observations. Exceeding it is a distinct
    failure from running out of steps and needs a distinct mitigation.
``CostLedger`` budget
    Money. Steps and tokens are not proportional -- a bloated observation
    costs many tokens in one step -- so a step cap does not bound spend.

Token accounting uses ``llm_client_kit.cost.CostLedger`` with a synthetic
price table. The token counts are *simulated* (a fixed cost per step plus the
observation's estimated length), so the dollar figures are not a forecast of
anyone's bill. What they are is a consistent unit for comparing topologies,
which is the comparison being made.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from llm_client_kit.cost import CostLedger, ModelPrice

from .faults import FaultConfig, FaultInjector
from .server import MCPServer

# A synthetic model identity. Named to be unmistakable in any output: nobody
# should be able to read a results table and think this was a real provider.
SIM_MODEL = "sim-agent-v1"
SIM_PRICES = {SIM_MODEL: ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60)}

# Tokens charged per reasoning step, before observations. Fixed so that step
# count and token count are related but not identical -- which is exactly the
# relationship that makes "supervisor takes more steps" and "supervisor costs
# more tokens" two separate claims.
TOKENS_PER_STEP_PROMPT = 320
TOKENS_PER_STEP_COMPLETION = 60
# Every worker/supervisor handoff re-states the task and the tool list to a
# fresh context. This is the token tax that the supervisor question is about.
TOKENS_PER_HANDOFF = 450


class Outcome(str, Enum):
    """Terminal state of one agent run."""

    SUCCESS = "success"
    STEP_CAP = "step_cap"                # infinite loop, caught by the cap
    CONTEXT_EXHAUSTED = "context_exhausted"
    TOOL_ERROR_CASCADE = "error_cascade"
    WRONG_ANSWER = "wrong_answer"
    PARTIAL_ROLLBACK = "partial_rollback"


FAILURE_OUTCOMES = frozenset(o for o in Outcome if o is not Outcome.SUCCESS)


@dataclass(frozen=True)
class Task:
    """One item of work, with a checkable expected answer.

    ``expected`` is a substring the final answer must contain. Substring
    rather than exact match because the topologies phrase their answers
    differently, and the question under test is whether the agent got the
    *fact* right, not whether it formatted it identically.
    """

    id: str
    goal: str
    plan: tuple[tuple[str, dict[str, Any]], ...]
    expected: str


def _ids_for(owner: str) -> list[str]:
    from .tools import SEED_RECORDS

    return sorted(k for k, v in SEED_RECORDS.items() if v["owner"] == owner)


TASKS: tuple[Task, ...] = (
    Task(
        id="t1-audit-ops",
        goal="Total the amounts of every record owned by ops.",
        plan=(
            ("search_records", {"query": "ops"}),
            ("fetch_record", {"record_id": "rec-001"}),
            ("fetch_record", {"record_id": "rec-004"}),
            ("summarise_amounts", {"record_ids": _ids_for("ops")}),
        ),
        expected="total=4570",
    ),
    Task(
        id="t2-audit-sre",
        goal="Total the amounts of every record owned by sre.",
        plan=(
            ("search_records", {"query": "sre"}),
            ("fetch_record", {"record_id": "rec-003"}),
            ("summarise_amounts", {"record_ids": _ids_for("sre")}),
        ),
        expected="total=0",
    ),
    Task(
        id="t3-finance-tag",
        goal="Total the amounts of every record tagged finance.",
        plan=(
            ("search_records", {"query": "finance"}),
            ("fetch_record", {"record_id": "rec-002"}),
            ("summarise_amounts", {"record_ids": ["rec-001", "rec-002", "rec-004"]}),
        ),
        expected="total=14370",
    ),
    Task(
        id="t4-note-and-total",
        goal="Record a note about the vendor renewal, then total all finance records.",
        plan=(
            ("search_records", {"query": "vendor"}),
            ("fetch_record", {"record_id": "rec-004"}),
            ("append_note", {"text": "vendor renewal reviewed"}),
            ("summarise_amounts", {"record_ids": ["rec-001", "rec-002", "rec-004"]}),
        ),
        expected="total=14370",
    ),
    Task(
        id="t5-gated-delete",
        goal="Delete the access review record, honouring the confirmation gate.",
        plan=(
            ("search_records", {"query": "security"}),
            ("delete_record", {"record_id": "rec-005"}),
            ("delete_record", {"record_id": "rec-005", "confirm_token": "AUTO"}),
        ),
        expected="deleted rec-005",
    ),
    Task(
        id="t6-cross-owner",
        goal="Total the amounts of the two largest records.",
        plan=(
            ("search_records", {"query": "finance"}),
            ("fetch_record", {"record_id": "rec-002"}),
            ("fetch_record", {"record_id": "rec-004"}),
            ("summarise_amounts", {"record_ids": ["rec-002", "rec-004"]}),
        ),
        expected="total=12950",
    ),
)


@dataclass
class StepRecord:
    """One executed step, kept for the trace that the taxonomy cites."""

    index: int
    tool: str
    args: dict[str, Any]
    faults: list[str]
    is_error: bool
    observation: str


@dataclass
class RunResult:
    """The outcome of one agent run over one task."""

    task_id: str
    topology: str
    outcome: Outcome
    steps: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    handoffs: int
    trace: list[StepRecord] = field(default_factory=list)
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.outcome in FAILURE_OUTCOMES

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def estimate_tokens(text: str) -> int:
    """Length/4, the standard rough proxy.

    Deliberately crude and deliberately deterministic. A real tokenizer would
    add a dependency and change nothing about the comparison, because every
    topology is charged by the same rule.
    """
    return max(1, len(text) // 4)


@dataclass
class AgentConfig:
    max_steps: int = 12
    context_limit: int = 3000
    # Consecutive tool errors before the run is declared a cascade. Two is
    # too twitchy -- one bad arg followed by one recovery attempt is normal.
    # Three consecutive failures means the agent is not reading the errors.
    cascade_threshold: int = 3
    faults: FaultConfig = field(default_factory=FaultConfig)


class ScriptedAgent:
    """A deterministic plan-and-execute policy over an MCP session.

    Not a model. See the module docstring.
    """

    def __init__(
        self,
        server: MCPServer,
        config: AgentConfig,
        injector: FaultInjector,
        ledger: CostLedger,
        *,
        label: str = "agent",
    ) -> None:
        self.server = server
        self.config = config
        self.injector = injector
        self.ledger = ledger
        self.label = label
        self.tool_names = self._discover_tools()

    def _discover_tools(self) -> list[str]:
        """Read the tool list off the wire, as a real client must.

        Hardcoding the tool names would make the agent depend on the server's
        source rather than on its protocol, and would quietly pass a test
        that a real client would fail.
        """
        response = self.server.handle_request(
            _req("tools/list", {}, id_=f"{self.label}-list")
        )
        assert response is not None and "result" in response, response
        return [t["name"] for t in response["result"]["tools"]]

    # --- the loop ------------------------------------------------------

    def run(
        self,
        task: Task,
        plan: Sequence[tuple[str, dict[str, Any]]] | None = None,
        *,
        budget_steps: int | None = None,
        context_used: int = 0,
    ) -> tuple[Outcome, list[StepRecord], int, int, str]:
        """Execute a plan. Returns (outcome, trace, steps, context_used, detail).

        ``context_used`` is threaded in and out rather than owned so a
        pipeline stage can inherit the context its predecessor consumed --
        which is the mechanism that makes context exhaustion a *topology*
        property rather than a per-agent one.
        """
        steps_allowed = budget_steps if budget_steps is not None else self.config.max_steps
        remaining = list(plan if plan is not None else task.plan)
        trace: list[StepRecord] = []
        consecutive_errors = 0
        steps = 0
        last_action: tuple[str, str] | None = None
        answer_seen = False

        while steps < steps_allowed:
            if not remaining:
                break

            steps += 1
            faults = self.injector.draw_all()
            fired = [k for k, v in faults.items() if v]

            tool, args = remaining[0]
            if tool == "delete_record" and args.get("confirm_token") == "AUTO":
                # The gate is honoured, not bypassed: the token comes from
                # what the previous call actually returned.
                args = dict(args)
                args["confirm_token"] = self.server.ctx.confirm_token(args["record_id"])

            # --- fault application, in a fixed order -------------------
            if faults["stall"]:
                # Repeat the previous action instead of advancing. This is the
                # infinite loop: no progress, and without max_steps it never
                # terminates.
                if last_action is not None:
                    tool, args = last_action[0], dict(remaining[0][1])
                    trace.append(
                        StepRecord(steps, tool, args, fired, False, "(stalled: repeated action)")
                    )
                    self._charge("(stalled: repeated action)")
                    continue

            if faults["misselect"]:
                tool = self._neighbour_of(tool)

            call_args = dict(args)
            if faults["bad_args"]:
                call_args = self._corrupt(call_args)

            if faults["transient"]:
                # The call never reached the server. Modelled as a failed step
                # rather than as a silent retry because that is what the agent
                # sees: an error observation it must decide how to handle. The
                # retry policy below is what turns it into a recoverable one.
                observation, is_error = (
                    "transient transport failure: connection reset before response. "
                    "The call did not reach the server; retrying is safe.",
                    True,
                )
            else:
                observation, is_error = self._call(tool, call_args)

            if faults["context_bloat"]:
                # A tool that returned far more than was asked for. The token
                # cost is real even though the useful content did not grow.
                observation = observation + " " + ("padding " * 120)

            self._charge(observation)
            context_used += estimate_tokens(observation)
            trace.append(StepRecord(steps, tool, call_args, fired, is_error, observation[:200]))
            last_action = (tool, str(call_args))

            if context_used > self.config.context_limit:
                return (
                    Outcome.CONTEXT_EXHAUSTED,
                    trace,
                    steps,
                    context_used,
                    f"context {context_used} > limit {self.config.context_limit}",
                )

            # A confirmation gate firing is the gate working, not a fault. The
            # tool correctly refused and told us how to proceed, and the plan
            # already anticipates a second call carrying the token. Counting
            # it as a tool error would make every gated task look like a
            # cascade -- which it did, until this was fixed.
            gated = is_error and "confirmation required" in observation
            if gated:
                remaining.pop(0)
                consecutive_errors = 0
                continue

            if is_error:
                consecutive_errors += 1
                if consecutive_errors >= self.config.cascade_threshold:
                    return (
                        Outcome.TOOL_ERROR_CASCADE,
                        trace,
                        steps,
                        context_used,
                        f"{consecutive_errors} consecutive tool errors",
                    )
                # The error observation names the fix. Whether the policy acts
                # on it is the competence draw -- which is the whole of what
                # "a better model" means in this simulation.
                if not self.injector.recovers():
                    continue
                # Recovered: retry the same intended step cleanly next turn.
                continue

            consecutive_errors = 0
            if tool == remaining[0][0]:
                completed = remaining.pop(0)
                if completed[0] in {"summarise_amounts", "delete_record"}:
                    if task.expected in observation:
                        answer_seen = True
        else:
            # while-else: the step cap was hit with plan steps outstanding.
            return (
                Outcome.STEP_CAP,
                trace,
                steps,
                context_used,
                f"hit step cap {steps_allowed} with {len(remaining)} steps outstanding",
            )

        if remaining:
            return (
                Outcome.STEP_CAP,
                trace,
                steps,
                context_used,
                f"plan not exhausted: {len(remaining)} steps outstanding",
            )
        if not answer_seen:
            return (
                Outcome.WRONG_ANSWER,
                trace,
                steps,
                context_used,
                f"plan completed but expected {task.expected!r} never observed",
            )
        return Outcome.SUCCESS, trace, steps, context_used, ""

    # --- helpers -------------------------------------------------------

    def _call(self, tool: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Invoke a tool over the protocol and flatten the response.

        A JSON-RPC error and a tool error are both surfaced to the policy as
        (text, is_error=True), because from the agent's seat both mean "that
        did not work". They are kept distinct on the wire -- which is what
        lets the conformance tests check the distinction the agent flattens.
        """
        response = self.server.handle_request(
            _req("tools/call", {"name": tool, "arguments": args}, id_=f"{self.label}-{tool}")
        )
        assert response is not None
        if "error" in response:
            err = response["error"]
            return f"protocol error {err['code']}: {err.get('data', err['message'])}", True
        result = response["result"]
        text = " ".join(c["text"] for c in result["content"])
        return text, bool(result.get("isError"))

    def _neighbour_of(self, tool: str) -> str:
        """The plausible-but-wrong tool. Not a uniform random pick.

        Misselection in practice is not uniform: an agent reaches for the tool
        whose description sounds nearest, which is why the two record-lookup
        tools are the confusable pair. Drawing uniformly would model a
        different, much less interesting failure.
        """
        neighbours = {
            "search_records": "fetch_record",
            "fetch_record": "search_records",
            "summarise_amounts": "fetch_record",
            "append_note": "search_records",
            "delete_record": "fetch_record",
        }
        candidate = neighbours.get(tool, tool)
        return candidate if candidate in self.tool_names else tool

    def _corrupt(self, args: dict[str, Any]) -> dict[str, Any]:
        """Malform one argument in a way the schema will reject."""
        if not args:
            return args
        out = dict(args)
        key = sorted(out)[0]
        value = out[key]
        out[key] = 12345 if isinstance(value, str) else "not-a-list"
        return out

    def _charge(self, observation: str) -> None:
        self.ledger.record(
            SIM_MODEL,
            prompt_tokens=TOKENS_PER_STEP_PROMPT + estimate_tokens(observation),
            completion_tokens=TOKENS_PER_STEP_COMPLETION,
        )

    def charge_handoff(self) -> None:
        """The cost of re-establishing context in another agent.

        Charged by the topologies, not by the loop, because it is a property
        of how work is routed rather than of any single agent's reasoning.
        """
        self.ledger.record(
            SIM_MODEL, prompt_tokens=TOKENS_PER_HANDOFF, completion_tokens=40
        )


def _req(method: str, params: dict[str, Any], *, id_: Any, notify: bool = False):
    from .protocol import Request

    return Request(method=method, params=params, id=id_, is_notification=notify)


def new_session() -> MCPServer:
    """A freshly initialized MCP session.

    Every run gets its own. Sharing a session across runs would let one run's
    deletions change the next run's task, so the measured failure rate would
    depend on execution order.
    """
    server = MCPServer()
    server.handle_request(
        _req("initialize", {"protocolVersion": "2025-06-18"}, id_="init")
    )
    server.handle_request(_req("notifications/initialized", {}, id_=None, notify=True))
    return server


def new_ledger(budget_usd: float | None = None) -> CostLedger:
    return CostLedger(prices=SIM_PRICES, budget_usd=budget_usd)

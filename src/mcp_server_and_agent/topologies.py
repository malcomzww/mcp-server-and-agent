"""Four topologies over the same task set, the same server, the same faults.

The comparison is only meaningful if the *only* thing that varies is how work
is routed. So every topology here gets: a fresh MCP session seeded identically,
the same ``FaultConfig``, and a seed derived from ``(topology, task, trial)``.
What differs is the number of agents, how the plan is divided, and what each
agent carries into its context.

The four:

``single``
    One agent, the whole tool list, the whole plan. The baseline. Fewest
    handoffs, so the fewest tokens, and one context that accumulates
    everything -- so it is the topology most exposed to context exhaustion.

``supervisor``
    A supervisor decomposes the plan and dispatches each step to a fresh
    worker, then aggregates. Each dispatch pays ``TOKENS_PER_HANDOFF`` because
    the worker starts cold and must be told the task and the tools. The
    hypothesis under test is that this tax buys nothing.

``pipeline``
    A fixed sequential chain: discover -> fetch -> aggregate. No routing
    decision at all, so no routing to get wrong, but also no ability to
    re-plan when a stage fails. Context is threaded forward, so a stage
    inherits its predecessor's consumption.

``reflexive``
    Single agent plus a reflection pass: on failure it inspects its own trace
    and retries once with a shortened plan. Chosen as the fourth because it
    isolates one variable -- whether a *second attempt* is worth more than a
    *second agent*. Supervisor and reflexive both cost extra tokens; only one
    of them adds a genuinely new capability.

The isolation each topology gives is the load-bearing difference. A fresh
worker per step means an error at step k does not poison step k+1's context --
which should suppress error cascades. Whether that suppression is worth the
token tax is the measured question, not an assumption.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from llm_client_kit.cost import CostLedger

from .agent import (
    AgentConfig,
    Outcome,
    RunResult,
    ScriptedAgent,
    StepRecord,
    Task,
    new_ledger,
    new_session,
)
from .faults import FaultInjector, run_seed

TOPOLOGIES = ("single", "supervisor", "pipeline", "reflexive")


@dataclass
class _Run:
    """Shared setup for one (topology, task, trial) cell."""

    task: Task
    config: AgentConfig
    seed: int
    ledger: CostLedger

    def agent(self, label: str, injector: FaultInjector) -> ScriptedAgent:
        return ScriptedAgent(new_session(), self.config, injector, self.ledger, label=label)


def _finish(
    topology: str,
    run: _Run,
    outcome: Outcome,
    trace: list[StepRecord],
    steps: int,
    handoffs: int,
    detail: str,
) -> RunResult:
    summary = run.ledger.summary()
    return RunResult(
        task_id=run.task.id,
        topology=topology,
        outcome=outcome,
        steps=steps,
        prompt_tokens=int(summary["prompt_tokens"]),
        completion_tokens=int(summary["completion_tokens"]),
        cost_usd=float(summary["total_usd"]),
        handoffs=handoffs,
        trace=trace,
        detail=detail,
    )


def run_single(run: _Run) -> RunResult:
    """One agent, one context, the whole plan."""
    injector = FaultInjector.seeded(run.config.faults, run.seed)
    agent = run.agent("single", injector)
    outcome, trace, steps, _ctx, detail = agent.run(run.task)
    return _finish("single", run, outcome, trace, steps, 0, detail)


def run_supervisor(run: _Run) -> RunResult:
    """A supervisor dispatching each plan step to a fresh worker.

    Each worker gets its own MCP session and its own clean context, but they
    share the fault injector: the faults are a property of the *run*, not of
    each agent, or a topology that spawns more agents would draw more faults
    and the comparison would be rigged in favour of whoever spawns fewest.

    They share the ledger too, which is the point -- the supervisor's cost is
    the sum of its workers plus the handoff tax.
    """
    injector = FaultInjector.seeded(run.config.faults, run.seed)
    supervisor = run.agent("supervisor", injector)

    trace: list[StepRecord] = []
    total_steps = 0
    handoffs = 0

    for index, step in enumerate(run.task.plan):
        # The dispatch itself costs tokens: the worker is cold and must be
        # briefed. This is charged before any work happens, which is why a
        # supervisor that dispatches and then fails still costs more.
        supervisor.charge_handoff()
        handoffs += 1

        worker = run.agent(f"worker-{index}", injector)
        # Worker state must match what the plan assumes at this point, so
        # earlier mutating steps are replayed into the worker's fresh session.
        _replay_prefix(worker, run.task, index)

        # Per-worker step budget. A worker that loops does not consume the
        # whole run's budget -- containment is the supervisor's actual
        # selling point, so it must be implemented rather than assumed.
        outcome, wtrace, wsteps, _ctx, detail = worker.run(
            run.task, plan=[step], budget_steps=max(2, run.config.max_steps // 2)
        )
        trace.extend(wtrace)
        total_steps += wsteps

        if outcome is not Outcome.SUCCESS and outcome is not Outcome.WRONG_ANSWER:
            return _finish("supervisor", run, outcome, trace, total_steps, handoffs, detail)

    # Aggregation pass: the supervisor re-reads what the workers returned.
    supervisor.charge_handoff()
    handoffs += 1
    if not _answer_in(trace, run.task.expected):
        return _finish(
            "supervisor", run, Outcome.WRONG_ANSWER, trace, total_steps, handoffs,
            f"workers completed but expected {run.task.expected!r} never observed",
        )
    return _finish("supervisor", run, Outcome.SUCCESS, trace, total_steps, handoffs, "")


def run_pipeline(run: _Run) -> RunResult:
    """A fixed three-stage chain with context threaded forward.

    Stages are cut by tool kind rather than by step count so the split means
    something: discovery, retrieval, aggregation. A stage failure ends the
    run -- there is no re-planner, which is the trade the topology makes.
    """
    injector = FaultInjector.seeded(run.config.faults, run.seed)
    stages = _stage_split(run.task)

    trace: list[StepRecord] = []
    total_steps = 0
    handoffs = 0
    context_used = 0

    for index, stage_plan in enumerate(stages):
        if not stage_plan:
            continue
        stage = run.agent(f"stage-{index}", injector)
        _replay_prefix(stage, run.task, _plan_index_of(run.task, stage_plan[0]))
        if index > 0:
            stage.charge_handoff()
            handoffs += 1
        outcome, strace, ssteps, context_used, detail = stage.run(
            run.task, plan=list(stage_plan), context_used=context_used
        )
        trace.extend(strace)
        total_steps += ssteps
        if outcome is not Outcome.SUCCESS and outcome is not Outcome.WRONG_ANSWER:
            return _finish("pipeline", run, outcome, trace, total_steps, handoffs, detail)

    if not _answer_in(trace, run.task.expected):
        return _finish(
            "pipeline", run, Outcome.WRONG_ANSWER, trace, total_steps, handoffs,
            f"pipeline completed but expected {run.task.expected!r} never observed",
        )
    return _finish("pipeline", run, Outcome.SUCCESS, trace, total_steps, handoffs, "")


def run_reflexive(run: _Run) -> RunResult:
    """Single agent, plus exactly one reflection-and-retry on failure.

    The retry is not a blind rerun. It inspects the trace, drops the steps
    that already succeeded, and re-attempts only what is outstanding -- which
    is why it can beat a rerun on both cost and outcome. One retry, not a
    loop: an unbounded reflect-retry is itself an infinite-loop generator, and
    this repo is supposed to be measuring those, not shipping one.
    """
    injector = FaultInjector.seeded(run.config.faults, run.seed)
    agent = run.agent("reflexive", injector)
    outcome, trace, steps, _ctx, detail = agent.run(run.task)

    if outcome is Outcome.SUCCESS:
        return _finish("reflexive", run, outcome, trace, steps, 0, detail)

    # Reflection: what did we already get done?
    done = {s.tool for s in trace if not s.is_error}
    retry_plan = [(t, a) for (t, a) in run.task.plan if t not in done or t == "summarise_amounts"]
    if not retry_plan:
        retry_plan = list(run.task.plan)

    retry = run.agent("reflexive-retry", injector)
    retry.charge_handoff()
    _replay_prefix(retry, run.task, _plan_index_of(run.task, retry_plan[0]))
    outcome2, trace2, steps2, _ctx2, detail2 = retry.run(run.task, plan=retry_plan)
    trace = trace + trace2
    return _finish(
        "reflexive", run, outcome2, trace, steps + steps2, 1,
        detail2 or f"retry after {outcome.value}",
    )


# --- shared helpers ----------------------------------------------------


def _answer_in(trace: list[StepRecord], expected: str) -> bool:
    return any(expected in s.observation for s in trace if not s.is_error)


def _plan_index_of(task: Task, step: tuple) -> int:
    for i, s in enumerate(task.plan):
        if s[0] == step[0] and s[1] == step[1]:
            return i
    return 0


def _replay_prefix(agent: ScriptedAgent, task: Task, upto: int) -> None:
    """Bring a fresh session's *state* up to the point in the plan we resume at.

    Only mutating tools are replayed, and they are replayed directly against
    the server without going through the policy -- so the replay costs no
    tokens and draws no faults. It exists to make multi-session topologies
    comparable to single-session ones, not to give them free work: a
    supervisor whose workers each started from an empty store would be
    solving a different, easier task.
    """
    from .tools import TOOLS

    for tool, args in task.plan[:upto]:
        if TOOLS[tool].destructive or not TOOLS[tool].idempotent:
            call = dict(args)
            if tool == "delete_record" and call.get("confirm_token") == "AUTO":
                call["confirm_token"] = agent.server.ctx.confirm_token(call["record_id"])
            TOOLS[tool].handler(agent.server.ctx, call)


def _stage_split(task: Task) -> list[list[tuple]]:
    """Cut a plan into discover / retrieve / aggregate stages."""
    discover = [s for s in task.plan if s[0] == "search_records"]
    retrieve = [s for s in task.plan if s[0] in {"fetch_record", "append_note"}]
    aggregate = [s for s in task.plan if s[0] in {"summarise_amounts", "delete_record"}]
    return [discover, retrieve, aggregate]


RUNNERS: dict[str, Callable[[_Run], RunResult]] = {
    "single": run_single,
    "supervisor": run_supervisor,
    "pipeline": run_pipeline,
    "reflexive": run_reflexive,
}


def run_one(
    topology: str, task: Task, trial: int, config: AgentConfig, base_seed: int
) -> RunResult:
    """Run one cell of the experiment grid."""
    if topology not in RUNNERS:
        raise ValueError(f"unknown topology {topology!r}; known: {sorted(RUNNERS)}")
    # The seed deliberately does NOT include the topology, so every topology
    # faces the identical fault stream for a given (task, trial). Paired
    # comparison, not two independent samples -- which is what lets a
    # difference between topologies be attributed to the topology.
    seed = run_seed("shared", task.id, trial, base_seed)
    run = _Run(task=task, config=config, seed=seed, ledger=new_ledger())
    return RUNNERS[topology](run)


def run_grid(
    tasks: tuple[Task, ...], config: AgentConfig, *, trials: int, base_seed: int
) -> list[RunResult]:
    """Every topology x task x trial. The full experiment."""
    out: list[RunResult] = []
    for topology in TOPOLOGIES:
        for task in tasks:
            for trial in range(trials):
                out.append(run_one(topology, task, trial, config, base_seed))
    return out

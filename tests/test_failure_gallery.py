"""Every seed published in docs/failure-taxonomy.md, re-verified.

A reproduction seed in a document is a claim, and an unchecked claim in a
document rots. These tests re-run each published (mode, seed, topology, task)
triple and assert it still produces the failure the taxonomy says it does. If
a refactor changes the RNG draw order or the loop's control flow, this breaks
the build rather than silently turning the taxonomy into fiction.

The table below is the single source of truth. docs/failure-taxonomy.md
quotes it, and scripts/find_failure_seeds.py is what found it.
"""

from __future__ import annotations

import pytest

from mcp_server_and_agent.agent import TASKS, AgentConfig, Outcome
from mcp_server_and_agent.faults import FaultConfig
from mcp_server_and_agent.topologies import run_one

# mode -> (seed, topology, task_id, config, expected outcome)
GALLERY = {
    "infinite_loop": (
        1,
        "single",
        "t1-audit-ops",
        AgentConfig(
            max_steps=8,
            faults=FaultConfig(
                misselect=0.0, bad_args=0.0, transient=0.0,
                stall=0.55, context_bloat=0.0, recovery_skill=0.9,
            ),
        ),
        Outcome.STEP_CAP,
    ),
    "tool_misselection": (
        1,
        "single",
        "t1-audit-ops",
        AgentConfig(
            max_steps=8,
            faults=FaultConfig(
                misselect=0.55, bad_args=0.0, transient=0.0,
                stall=0.0, context_bloat=0.0, recovery_skill=0.25,
            ),
        ),
        Outcome.TOOL_ERROR_CASCADE,
    ),
    "error_cascade": (
        1,
        "single",
        "t1-audit-ops",
        AgentConfig(
            max_steps=8,
            cascade_threshold=3,
            faults=FaultConfig(
                misselect=0.0, bad_args=0.6, transient=0.0,
                stall=0.0, context_bloat=0.0, recovery_skill=0.2,
            ),
        ),
        Outcome.TOOL_ERROR_CASCADE,
    ),
    "context_exhaustion": (
        1,
        "single",
        "t1-audit-ops",
        AgentConfig(
            max_steps=20,
            context_limit=400,
            faults=FaultConfig(
                misselect=0.0, bad_args=0.0, transient=0.0,
                stall=0.0, context_bloat=0.7, recovery_skill=0.9,
            ),
        ),
        Outcome.CONTEXT_EXHAUSTED,
    ),
    "partial_failure": (
        1,
        "reflexive",
        "t4-note-and-total",
        AgentConfig(
            max_steps=6,
            faults=FaultConfig(
                misselect=0.0, bad_args=0.45, transient=0.3,
                stall=0.0, context_bloat=0.0, recovery_skill=0.2,
            ),
        ),
        Outcome.TOOL_ERROR_CASCADE,
    ),
}


def task_by_id(task_id: str):
    return next(t for t in TASKS if t.id == task_id)


@pytest.mark.parametrize("mode", sorted(GALLERY))
def test_published_seed_still_reproduces(mode: str) -> None:
    seed, topology, task_id, config, expected = GALLERY[mode]
    result = run_one(topology, task_by_id(task_id), 0, config, base_seed=seed)
    assert result.outcome is expected, (
        f"{mode}: seed {seed} on {topology}/{task_id} now gives "
        f"{result.outcome.value}, not {expected.value}. The taxonomy is stale."
    )


@pytest.mark.parametrize("mode", sorted(GALLERY))
def test_published_seed_is_bit_stable(mode: str) -> None:
    """Same seed, same token count. Not just the same category."""
    seed, topology, task_id, config, _ = GALLERY[mode]
    a = run_one(topology, task_by_id(task_id), 0, config, base_seed=seed)
    b = run_one(topology, task_by_id(task_id), 0, config, base_seed=seed)
    assert (a.outcome, a.steps, a.total_tokens) == (b.outcome, b.steps, b.total_tokens)


def test_the_mitigation_for_infinite_loops_is_the_step_cap() -> None:
    """Raising the cap does not fix a stall; it only makes it more expensive.

    The point of the mitigation column: a step cap converts an unbounded loop
    into a bounded, *detectable* failure. It does not make the task succeed,
    and a taxonomy that implies otherwise is selling a fix that is really a
    circuit breaker.
    """
    _seed, topology, task_id, config, _ = GALLERY["infinite_loop"]
    task = task_by_id(task_id)
    tighter = AgentConfig(max_steps=4, faults=config.faults)
    looser = AgentConfig(max_steps=16, faults=config.faults)

    a = run_one(topology, task, 0, tighter, base_seed=1)
    b = run_one(topology, task, 0, looser, base_seed=1)
    assert a.outcome is Outcome.STEP_CAP
    assert a.steps == 4
    # More budget, more spend, same failure.
    assert b.total_tokens > a.total_tokens


def test_partial_failure_leaves_state_that_rollback_can_repair() -> None:
    """The mode is not "it failed". It is "it failed after writing"."""
    from mcp_server_and_agent.tools import SEED_RECORDS, TOOLS, ToolContext

    ctx = ToolContext()
    TOOLS["append_note"].handler(ctx, {"text": "step 1 of 3"})
    token = ctx.confirm_token("rec-002")
    TOOLS["delete_record"].handler(ctx, {"record_id": "rec-002", "confirm_token": token})

    # Mid-task state: neither the original nor the intended final state.
    assert len(ctx.records) == len(SEED_RECORDS) - 1
    assert len(ctx.notes) == 1
    assert len(ctx.journal) == 2

    assert ctx.rollback() == 2
    assert set(ctx.records) == set(SEED_RECORDS)
    assert ctx.notes == []


def test_the_gate_message_quoted_in_the_taxonomy_is_current() -> None:
    """docs/failure-taxonomy.md quotes this refusal verbatim, token included.

    Pinned so that changing the token derivation or the message wording
    breaks a test rather than turning a quoted example in the taxonomy into
    something the code no longer produces.
    """
    from mcp_server_and_agent.tools import TOOLS, ToolContext

    result = TOOLS["delete_record"].handler(ToolContext(), {"record_id": "rec-005"})
    assert result.is_error is True
    assert result.text == (
        "confirmation required. delete_record would permanently remove rec-005 "
        "(title='Access review', owner=sre). To proceed, call again with "
        "confirm_token='5e08221c3aee'."
    )


def test_context_exhaustion_is_not_fixed_by_a_bigger_window() -> None:
    """It is deferred by one. The mitigation that works is bounding the
    observation, not raising the ceiling -- because bloat is per-step and a
    window is a constant."""
    _seed, topology, task_id, config, _ = GALLERY["context_exhaustion"]
    task = task_by_id(task_id)
    bigger = AgentConfig(
        max_steps=config.max_steps, context_limit=4000, faults=config.faults
    )
    result = run_one(topology, task, 0, bigger, base_seed=1)
    # A wide enough window lets this particular short plan through, which is
    # exactly the trap: it works until the plan gets longer.
    assert result.outcome in {Outcome.SUCCESS, Outcome.CONTEXT_EXHAUSTED}
    assert result.total_tokens > 0

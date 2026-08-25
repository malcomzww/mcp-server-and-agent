"""Tests for the fault model, the topologies, and the determinism claim.

The determinism tests are the load-bearing ones. Everything committed under
results/ is guarded by a `git diff --exit-code` drift gate, and that gate is
only meaningful if the same seed genuinely produces the same numbers. A
non-deterministic experiment would make CI fail at random and train everyone
to ignore it.
"""

from __future__ import annotations

import pytest

from mcp_server_and_agent.agent import (
    TASKS,
    AgentConfig,
    Outcome,
    ScriptedAgent,
    estimate_tokens,
    new_ledger,
    new_session,
)
from mcp_server_and_agent.faults import CLEAN, FaultConfig, FaultInjector, hash_str, run_seed
from mcp_server_and_agent.topologies import TOPOLOGIES, run_grid, run_one

# --- the control -------------------------------------------------------


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_every_topology_solves_every_task_with_no_faults(topology, task) -> None:
    """The control condition. If a topology cannot complete a task with the
    fault rate set to zero, then its measured failure rate is reporting a bug
    in the harness rather than a property of the topology -- and every number
    downstream is meaningless."""
    result = run_one(topology, task, 0, AgentConfig(faults=CLEAN), base_seed=1)
    assert result.outcome is Outcome.SUCCESS, f"{topology}/{task.id}: {result.detail}"


def test_clean_runs_cost_the_fewest_tokens_for_single_agent() -> None:
    """Handoffs are a tax. With no faults there is nothing for extra agents
    to contain, so every multi-agent topology must cost strictly more."""
    cfg = AgentConfig(faults=CLEAN)
    costs = {
        topo: sum(run_one(topo, t, 0, cfg, base_seed=1).total_tokens for t in TASKS)
        for topo in TOPOLOGIES
    }
    assert costs["single"] < costs["supervisor"]
    assert costs["single"] < costs["pipeline"]
    # Reflexive only pays for a retry it never needs when nothing fails.
    assert costs["reflexive"] == costs["single"]


# --- determinism -------------------------------------------------------


def test_same_seed_gives_identical_results() -> None:
    cfg = AgentConfig(faults=FaultConfig())
    a = run_grid(TASKS, cfg, trials=3, base_seed=42)
    b = run_grid(TASKS, cfg, trials=3, base_seed=42)
    assert [(r.topology, r.task_id, r.outcome, r.total_tokens) for r in a] == [
        (r.topology, r.task_id, r.outcome, r.total_tokens) for r in b
    ]


def test_different_seeds_give_different_results() -> None:
    """A determinism test passes trivially if the seed is ignored."""
    cfg = AgentConfig(faults=FaultConfig().scaled(2.0))
    a = run_grid(TASKS, cfg, trials=3, base_seed=42)
    b = run_grid(TASKS, cfg, trials=3, base_seed=99)
    assert [r.outcome for r in a] != [r.outcome for r in b]


def test_string_hash_is_stable_across_processes() -> None:
    """Python's built-in hash is salted per process, so using it would
    destroy reproducibility across runs. This is the reason for hash_str."""
    assert hash_str("supervisor") == hash_str("supervisor")
    assert hash_str("single") != hash_str("supervisor")
    # Pinned values. A change to the hash silently renumbers every seed in the
    # failure taxonomy, so it must break a test rather than pass quietly.
    assert hash_str("shared") == 2767733972
    assert hash_str("t1-audit-ops") == 2004639925


def test_seed_derivation_is_a_function_of_run_identity() -> None:
    assert run_seed("a", "t1", 0, 7) == run_seed("a", "t1", 0, 7)
    assert run_seed("a", "t1", 0, 7) != run_seed("a", "t1", 1, 7)
    assert run_seed("a", "t1", 0, 7) != run_seed("b", "t1", 0, 7)


def test_all_topologies_face_the_identical_fault_stream() -> None:
    """The comparison is paired. If topologies drew different faults, a
    difference between them could be luck rather than routing."""
    # run_one derives its seed from ("shared", task, trial) precisely so the
    # topology name cannot enter it. Assert that by checking the fault stream
    # each topology actually saw is byte-identical.
    cfg = AgentConfig(faults=FaultConfig().scaled(2.0))
    streams = {}
    for topo in TOPOLOGIES:
        result = run_one(topo, TASKS[0], 0, cfg, base_seed=7)
        # The first step of every topology is the same plan step, so it must
        # have drawn the same faults.
        streams[topo] = result.trace[0].faults
    assert len(set(tuple(v) for v in streams.values())) == 1, streams


# --- the fault model ---------------------------------------------------


def test_every_fault_kind_is_drawn_on_every_step() -> None:
    """Short-circuiting would make the RNG stream depend on the path taken,
    so adding a fault kind would renumber every later draw and silently
    change historical results."""
    inj = FaultInjector.seeded(FaultConfig(), 5)
    before = inj.rng.random()
    inj2 = FaultInjector.seeded(FaultConfig(), 5)
    inj2.draw_all()
    # 5 kinds drawn means 5 values consumed, so the 6th differs from the 1st.
    assert inj2.rng.random() != before


def test_clean_config_never_fires() -> None:
    inj = FaultInjector.seeded(CLEAN, 3)
    for _ in range(200):
        assert not any(inj.draw_all().values())
    assert inj.drawn == []


def test_scaling_raises_every_rate_but_not_recovery() -> None:
    base = FaultConfig()
    scaled = base.scaled(2.0)
    assert scaled.misselect == pytest.approx(base.misselect * 2)
    assert scaled.transient == pytest.approx(base.transient * 2)
    assert scaled.recovery_skill == base.recovery_skill


def test_scaling_clamps_at_one() -> None:
    assert FaultConfig(misselect=0.6).scaled(10.0).misselect == 1.0


# --- failure detection -------------------------------------------------


def test_step_cap_terminates_a_stalling_agent() -> None:
    """With stall at 1.0 the agent makes no progress forever. The cap is the
    only thing between that and an infinite loop."""
    cfg = AgentConfig(
        max_steps=6,
        faults=FaultConfig(
            misselect=0, bad_args=0, transient=0, stall=1.0, context_bloat=0, recovery_skill=1.0
        ),
    )
    result = run_one("single", TASKS[0], 0, cfg, base_seed=1)
    assert result.outcome is Outcome.STEP_CAP
    assert result.steps == 6


def test_context_exhaustion_is_distinct_from_the_step_cap() -> None:
    """Two different failures needing two different mitigations. A step cap
    does not bound tokens: one bloated observation costs many tokens in a
    single step."""
    cfg = AgentConfig(
        max_steps=20,
        context_limit=200,
        faults=FaultConfig(
            misselect=0, bad_args=0, transient=0, stall=0, context_bloat=1.0, recovery_skill=1.0
        ),
    )
    result = run_one("single", TASKS[0], 0, cfg, base_seed=1)
    assert result.outcome is Outcome.CONTEXT_EXHAUSTED
    assert result.steps < cfg.max_steps


def test_error_cascade_fires_on_consecutive_failures() -> None:
    cfg = AgentConfig(
        cascade_threshold=3,
        faults=FaultConfig(
            misselect=0, bad_args=1.0, transient=0, stall=0, context_bloat=0, recovery_skill=1.0
        ),
    )
    result = run_one("single", TASKS[0], 0, cfg, base_seed=1)
    assert result.outcome is Outcome.TOOL_ERROR_CASCADE


def test_misselection_reaches_for_the_neighbouring_tool_not_a_random_one() -> None:
    """Real misselection is not uniform -- the agent reaches for the tool
    whose description sounds nearest."""
    agent = ScriptedAgent(
        new_session(), AgentConfig(), FaultInjector.seeded(CLEAN, 1), new_ledger()
    )
    assert agent._neighbour_of("search_records") == "fetch_record"
    assert agent._neighbour_of("fetch_record") == "search_records"


# --- rollback ----------------------------------------------------------


def test_rollback_undoes_a_partial_multi_step_write() -> None:
    """A task that dies midway has left writes behind. Without compensation
    the retry starts from a state nobody can describe."""
    from mcp_server_and_agent.tools import TOOLS, ToolContext

    ctx = ToolContext()
    TOOLS["append_note"].handler(ctx, {"text": "one"})
    token = ctx.confirm_token("rec-001")
    TOOLS["delete_record"].handler(ctx, {"record_id": "rec-001", "confirm_token": token})
    assert "rec-001" not in ctx.records
    assert len(ctx.notes) == 1

    undone = ctx.rollback()
    assert undone == 2
    assert "rec-001" in ctx.records
    assert ctx.notes == []
    assert ctx.journal == []


def test_rollback_is_idempotent() -> None:
    from mcp_server_and_agent.tools import ToolContext

    ctx = ToolContext()
    assert ctx.rollback() == 0
    assert ctx.rollback() == 0


# --- cost accounting ---------------------------------------------------


def test_handoffs_are_charged_and_counted() -> None:
    cfg = AgentConfig(faults=CLEAN)
    single = run_one("single", TASKS[0], 0, cfg, base_seed=1)
    supervisor = run_one("supervisor", TASKS[0], 0, cfg, base_seed=1)
    assert single.handoffs == 0
    assert supervisor.handoffs == len(TASKS[0].plan) + 1
    assert supervisor.total_tokens > single.total_tokens


def test_cost_is_positive_and_tracks_tokens() -> None:
    result = run_one("single", TASKS[0], 0, AgentConfig(faults=CLEAN), base_seed=1)
    assert result.cost_usd > 0
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens


def test_token_estimate_is_never_zero() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100


# --- discovery ---------------------------------------------------------


def test_agent_discovers_tools_over_the_protocol() -> None:
    """Hardcoding names would make the agent depend on the server's source
    rather than on its protocol."""
    agent = ScriptedAgent(
        new_session(), AgentConfig(), FaultInjector.seeded(CLEAN, 1), new_ledger()
    )
    assert "search_records" in agent.tool_names
    assert "delete_record" in agent.tool_names


def test_unknown_topology_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown topology"):
        run_one("mesh", TASKS[0], 0, AgentConfig(), base_seed=1)

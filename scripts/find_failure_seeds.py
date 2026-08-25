"""Search for a reproducing seed for each failure mode, and verify it.

The failure taxonomy is only worth reading if its seeds actually reproduce.
This script finds them by search rather than by assertion, prints them, and
re-runs each one to confirm the mode it claims. Its output is pasted into
docs/failure-taxonomy.md, and `tests/test_failure_gallery.py` re-verifies
every published seed on every CI run -- so a seed that stops reproducing
breaks the build instead of quietly becoming a lie in a document.

Run:  python scripts/find_failure_seeds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp_server_and_agent.agent import TASKS, AgentConfig, Outcome  # noqa: E402
from mcp_server_and_agent.faults import FaultConfig  # noqa: E402
from mcp_server_and_agent.topologies import TOPOLOGIES, run_one  # noqa: E402

# Each mode gets a fault configuration that makes it *reachable*, then the
# seed search finds a specific run that exhibits it. Isolating one fault at a
# time is the point: a seed found under all-faults-on does not demonstrate
# which fault caused the mode.
PROBES = {
    "infinite_loop": (
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
}

# The sixth mode does not fit the (config, outcome) shape: partial failure is
# not about *which* terminal outcome fires, it is about the state left behind
# when one does. It is searched for separately, over the tasks that write.
PARTIAL_FAILURE_CONFIG = AgentConfig(
    max_steps=6,
    faults=FaultConfig(
        misselect=0.0, bad_args=0.45, transient=0.3,
        stall=0.0, context_bloat=0.0, recovery_skill=0.2,
    ),
)
WRITING_TASKS = ("t4-note-and-total", "t5-gated-delete")


def search_partial_failure(limit: int = 4000):
    """A run that fails *after* mutating state, so a rollback is owed.

    The interesting case is not that the run failed. It is that the store no
    longer matches what either the agent or the operator believes, and the
    journal is the only thing that can put it back.
    """
    from mcp_server_and_agent.tools import SEED_RECORDS

    for seed in range(1, limit):
        for topo in TOPOLOGIES:
            for task_id in WRITING_TASKS:
                task = next(t for t in TASKS if t.id == task_id)
                result = run_one(topo, task, 0, PARTIAL_FAILURE_CONFIG, base_seed=seed)
                if not result.failed:
                    continue
                # Did it write before dying? Re-run against a session we hold
                # so the journal is inspectable.
                ctx = _replay_to_context(topo, task, seed)
                mutated = len(ctx.records) != len(SEED_RECORDS) or bool(ctx.notes)
                if mutated and ctx.journal:
                    return seed, topo, task_id, result, ctx
    return None


def _replay_to_context(topo: str, task, seed: int):
    """Re-run the same triple and hand back the server state it left."""
    # Patched on `topologies`, not on `agent`: topologies imports the name
    # directly, so rebinding it on the defining module would not intercept.
    import mcp_server_and_agent.topologies as topo_mod

    captured = []
    original = topo_mod.new_session

    def spy():
        server = original()
        captured.append(server)
        return server

    topo_mod.new_session = spy
    try:
        run_one(topo, task, 0, PARTIAL_FAILURE_CONFIG, base_seed=seed)
    finally:
        topo_mod.new_session = original
    # The last session is the one that was live when the run died.
    return captured[-1].ctx


def search(config: AgentConfig, want: Outcome, limit: int = 4000):
    """First (seed, topology, task) exhibiting `want`. None if not found."""
    for seed in range(1, limit):
        for topo in TOPOLOGIES:
            for task in TASKS:
                r = run_one(topo, task, 0, config, base_seed=seed)
                if r.outcome is want:
                    return seed, topo, task.id, r
    return None


def rate_per_topology(config: AgentConfig, want: Outcome, trials: int = 40) -> dict[str, float]:
    """How often each topology hits `want` under an isolated fault.

    This is the number the taxonomy reports per mode, and it is a different
    measurement from the headline table: there, all five faults fire at once
    and the modes blur together. Isolating one fault is what shows which
    topology is structurally protected against which failure.
    """
    from mcp_server_and_agent.topologies import run_grid

    rows = run_grid(TASKS, config, trials=trials, base_seed=20260825)
    out = {}
    for topo in TOPOLOGIES:
        rs = [r for r in rows if r.topology == topo]
        out[topo] = sum(1 for r in rs if r.outcome is want) / len(rs)
    return out


def main() -> None:
    for name, (config, want) in PROBES.items():
        found = search(config, want)
        if found is None:
            print(f"{name:22s} NOT FOUND (wanted {want.value})")
            continue
        seed, topo, task_id, result = found
        # Verify: rerun the exact triple and confirm it reproduces.
        again = run_one(topo, next(t for t in TASKS if t.id == task_id), 0, config, base_seed=seed)
        ok = again.outcome is want and again.total_tokens == result.total_tokens
        print(
            f"{name:22s} seed={seed:5d} topology={topo:11s} task={task_id:16s} "
            f"steps={result.steps:2d} tokens={result.total_tokens:5d} "
            f"reproduces={ok}"
        )
        print(f"{'':22s} detail: {result.detail}")
        rates = rate_per_topology(config, want)
        print(f"{'':22s} rate/topology: " + "  ".join(f"{k}={v:.1%}" for k, v in rates.items()))

    found = search_partial_failure()
    if found is None:
        print(f"{'partial_failure':22s} NOT FOUND")
        return
    seed, topo, task_id, result, ctx = found
    before = len(ctx.journal)
    undone = ctx.rollback()
    print(
        f"{'partial_failure':22s} seed={seed:5d} topology={topo:11s} task={task_id:16s} "
        f"outcome={result.outcome.value} journalled={before} rolled_back={undone}"
    )
    print(f"{'':22s} detail: {result.detail}")


if __name__ == "__main__":
    main()

"""The fault model. This is the experiment.

There is no API key in this environment and no live model anywhere in this
repo. That is not a limitation worked around -- it is the design. A topology's
failure rate measured against a live model is confounded by that model's
nondeterminism, its version, and the temperature you happened to set. Run it
twice and you get two numbers, and you cannot tell whether the difference came
from the topology or the sampler.

So the agent brain is a **deterministic scripted policy** (see :mod:`agent`)
driven by an explicit, parameterised fault process defined here. What gets
measured is real: given a stated per-step probability of each fault, how often
does each topology fail to complete the task, and what does it cost in tokens?
Those are properties of the *topology* -- of how many steps it takes, how it
routes work, and whether an error at step k propagates or is contained. The
fault model is the independent variable, and it is fully specified below.

The claim this supports is: "under fault model F, topology T fails at rate R."
The claim it does NOT support is: "GPT-x in a supervisor topology fails at
rate R." The README and docs/failure-taxonomy.md say so in those words.

Five injectable faults, each chosen because it maps to a real failure mode:

``misselect``
    The policy picks a plausible-but-wrong tool. Real cause: two tool
    descriptions that do not discriminate.
``bad_args``
    Right tool, malformed arguments. Real cause: schema the model half-read.
``transient``
    The call fails in a way a retry would fix. Real cause: the network.
``stall``
    The policy repeats its previous action, making no progress. Real cause:
    a model that cannot parse an error and re-emits its last plan.
``context_bloat``
    The step's observation is inflated. Real cause: a tool that returns a
    whole document when asked for a field. Drives context exhaustion.

Each is drawn from an independent Bernoulli per step, from a
``random.Random`` seeded per (topology, task, trial). Same seed in, same
faults out -- which is what makes the drift gate on results/ meaningful.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

FAULT_KINDS = ("misselect", "bad_args", "transient", "stall", "context_bloat")


@dataclass(frozen=True)
class FaultConfig:
    """Per-step fault probabilities.

    Defaults are the headline configuration. They are not calibrated against
    any real model -- calibrating them would require the live runs this repo
    deliberately does not do. They are set where the topologies separate:
    high enough that failures happen at a measurable rate within the step
    cap, low enough that most tasks still complete.
    """

    misselect: float = 0.08
    bad_args: float = 0.06
    transient: float = 0.10
    stall: float = 0.05
    context_bloat: float = 0.12

    # Whether the policy can recover from an error observation. A competent
    # agent reads the error text and adapts; an incompetent one does not.
    # This is what "competence" means operationally in this simulation.
    recovery_skill: float = 0.75

    def scaled(self, factor: float) -> FaultConfig:
        """All fault rates multiplied by `factor`, recovery held fixed.

        Used for the sensitivity sweep: a topology ranking that only holds at
        one fault rate is not a finding about topologies.
        """
        return replace(
            self,
            misselect=min(1.0, self.misselect * factor),
            bad_args=min(1.0, self.bad_args * factor),
            transient=min(1.0, self.transient * factor),
            stall=min(1.0, self.stall * factor),
            context_bloat=min(1.0, self.context_bloat * factor),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "misselect": self.misselect,
            "bad_args": self.bad_args,
            "transient": self.transient,
            "stall": self.stall,
            "context_bloat": self.context_bloat,
            "recovery_skill": self.recovery_skill,
        }


CLEAN = FaultConfig(
    misselect=0.0, bad_args=0.0, transient=0.0, stall=0.0, context_bloat=0.0,
    recovery_skill=1.0,
)


@dataclass
class FaultInjector:
    """Draws faults for one agent run. One instance per (topology, task, trial).

    Holds its own ``Random`` rather than touching the global one. A shared
    global RNG makes results depend on execution order, and the moment you
    add a topology the earlier numbers change -- which looks exactly like a
    regression and is not one.
    """

    config: FaultConfig
    rng: random.Random
    drawn: list[str] | None = None

    @classmethod
    def seeded(cls, config: FaultConfig, seed: int) -> FaultInjector:
        return cls(config=config, rng=random.Random(seed), drawn=[])

    def draw(self, kind: str) -> bool:
        """Bernoulli draw for one fault kind.

        Every kind is drawn on every step, in a fixed order, even when an
        earlier draw already fired. Short-circuiting would make the RNG
        stream depend on the path taken, so adding a fault kind would
        renumber every subsequent draw and silently change historical
        results.
        """
        p = getattr(self.config, kind)
        hit = self.rng.random() < p
        if hit and self.drawn is not None:
            self.drawn.append(kind)
        return hit

    def draw_all(self) -> dict[str, bool]:
        return {kind: self.draw(kind) for kind in FAULT_KINDS}

    def recovers(self) -> bool:
        """Does the policy successfully react to an error observation?"""
        return self.rng.random() < self.config.recovery_skill


def run_seed(topology: str, task_id: str, trial: int, base_seed: int) -> int:
    """Deterministic per-run seed.

    Derived from the run's identity rather than from a counter so that the
    seed for one cell of the results table can be recomputed in isolation --
    which is what makes "reproduction seed" in the failure taxonomy an
    actually usable instruction rather than a decoration.
    """
    h = (hash_str(topology) * 31 + hash_str(task_id)) * 31 + trial
    return (h + base_seed) % (2**31 - 1)


def hash_str(s: str) -> int:
    """Stable string hash.

    Python's built-in ``hash`` is salted per process (PYTHONHASHSEED), so
    using it here would make every run irreproducible across processes --
    which is the single easiest way to destroy a determinism claim.
    """
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h

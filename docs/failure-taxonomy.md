# Failure taxonomy: six modes, their seeds, their rates, their mitigations

**Dated 2026-08-25.** Regenerate the seeds with
`python scripts/find_failure_seeds.py`; every seed below is re-verified on
each CI run by `tests/test_failure_gallery.py`.

> **The agent brain is simulated.** No LLM is involved anywhere in this
> document. A deterministic scripted policy executes plans over the MCP
> server while a parameterised fault process perturbs its choices. Every rate
> below is a rate *under that fault model*. None of them is a claim about how
> often a real model does anything.

---

## Why isolate the faults

The headline table in `results/topologies.md` runs all five faults at once,
which is realistic and useless for diagnosis: when a run dies you cannot tell
which fault killed it, and every topology's failures blur into two buckets.

This document runs **one fault at a time, turned up high**. That is what
exposes the structure — and the structure is the finding. Under the blended
model the four topologies look similar (31.7% / 28.2% / 21.2% / 4.5%). Under
isolated faults they are radically different, and *which* topology helps
depends entirely on *which* fault you have.

That is the sentence this repo exists to support: **there is no best
topology, only a best topology per failure mode.** A supervisor is not
generically safer. It is specifically immune to two of these six and no help
at all on three others.

---

## 1. Infinite loop

The agent stops making progress and never terminates on its own.

**Mechanism.** The `stall` fault makes the policy re-emit its previous action
instead of advancing the plan. Nothing errors. Every step looks locally fine.

**Reproduction.** `seed=1`, topology `single`, task `t1-audit-ops`,
`stall=0.55`, `max_steps=8`, all other faults 0.

```
1. search_records   faults=-       matched record ids: rec-001, rec-003, rec-004, rec-005
2. search_records   faults=stall   (stalled: repeated action)
3. fetch_record     faults=-       rec-001: title='Ledger reconciliation' ...
4. fetch_record     faults=stall   (stalled: repeated action)
5. fetch_record     faults=stall   (stalled: repeated action)
6. fetch_record     faults=stall   (stalled: repeated action)
7. fetch_record     faults=-       rec-004: title='Vendor renewal' ...
8. fetch_record     faults=stall   (stalled: repeated action)
-> step_cap: hit step cap 8 with 1 steps outstanding
```

**Rate per topology** (`stall=0.55` isolated, 40 trials x 6 tasks):

| `single` | `supervisor` | `pipeline` | `reflexive` |
|---|---|---|---|
| 24.6% | **0.0%** | 1.2% | 1.2% |

**Detection.** A hard step cap. There is no cleverer detector available from
inside the loop — the whole problem is that a stalled agent's steps look
normal one at a time. Repetition detection on `(tool, args)` catches the
simple case and misses the two-step cycle.

**Mitigation.** *Per-unit step budgets, not a bigger global one.* This is why
supervisor scores 0.0% and single scores 24.6% on identical fault draws: the
supervisor gives each worker its own small budget, so one stalling step
cannot eat the whole run's allowance. Raising a single agent's cap does not
fix a stall — `tests/test_failure_gallery.py::test_the_mitigation_for_infinite_loops_is_the_step_cap`
asserts that a looser cap produces the same failure and spends more getting
there. **A step cap is a circuit breaker, not a fix.** It converts an
unbounded loss into a bounded, detectable one.

---

## 2. Tool misselection

The agent calls a plausible-but-wrong tool.

**Mechanism.** `search_records` and `fetch_record` are deliberately confusable
— both plausibly answer "get me the record about X". The `misselect` fault
swaps one for its nearest neighbour rather than for a random tool, because
that is how real misselection behaves.

**Reproduction.** `seed=1`, topology `single`, task `t1-audit-ops`,
`misselect=0.55`, `recovery_skill=0.25`.

```
1. fetch_record  faults=misselect  err=True  protocol error -32602: tool 'fetch_record' requires argument 'record_id'
2. fetch_record  faults=misselect  err=True  protocol error -32602: tool 'fetch_record' requires argument 'record_id'
3. fetch_record  faults=misselect  err=True  protocol error -32602: tool 'fetch_record' requires argument 'record_id'
-> error_cascade: 3 consecutive tool errors
```

**Rate per topology** (`misselect=0.55` isolated):

| `single` | `supervisor` | `pipeline` | `reflexive` |
|---|---|---|---|
| 40.8% | 44.2% | 44.2% | **15.0%** |

**Detection.** *The schema catches it, and that is the useful finding.* Note
the error code: `-32602`, a protocol error, not a tool error. The wrong tool
was handed the right tool's arguments and rejected them at validation, before
executing anything. A misselected tool with a *loose* schema would have run
and returned a plausible wrong answer — silent corruption instead of a loud
failure.

**Mitigation.** Two, in order of value:

1. **Tight required-argument schemas.** They convert misselection from a
   wrong answer into an immediate, named error. This costs nothing and is the
   single highest-leverage thing in this document.
2. **Fewer, better-discriminated tools.** The confusable pair exists here on
   purpose. In a real server, merging `search_records` and `fetch_record`
   into one tool with an optional `id` would remove the failure rather than
   detect it.

Note that no topology helps: supervisor and pipeline are *worse* than single
(44.2% vs 40.8%). Routing does not fix a bad tool list. Only `reflexive`
helps, and only because a second attempt gets a fresh draw.

---

## 3. Error cascade

One failure produces a run of failures the agent cannot break out of.

**Mechanism.** `bad_args` malforms an argument. The tool returns an error
naming the fix. With `recovery_skill=0.2` the policy usually fails to act on
it and repeats the same malformed call.

**Reproduction.** `seed=1`, topology `single`, task `t1-audit-ops`,
`bad_args=0.6`, `recovery_skill=0.2`, `cascade_threshold=3`.

```
1. search_records     faults=-         err=False matched record ids: rec-001, ...
2. fetch_record       faults=bad_args  err=True  -32602: argument 'record_id' must be string, got int
3. fetch_record       faults=bad_args  err=True  -32602: argument 'record_id' must be string, got int
4. fetch_record       faults=bad_args  err=True  -32602: argument 'record_id' must be string, got int
-> error_cascade: 3 consecutive tool errors
```

**Rate per topology** (`bad_args=0.6` isolated):

| `single` | `supervisor` | `pipeline` | `reflexive` |
|---|---|---|---|
| 53.8% | 56.7% | 56.7% | **21.7%** |

**Detection.** A consecutive-error counter, tripped at 3. Two is too twitchy
— one bad call plus one recovery attempt is normal behaviour, and a detector
that fires on normal behaviour gets disabled.

**Mitigation.** **Errors written as model feedback.** Every tool here returns
what was wrong *and what would work instead*: `fetch_record` on a bad id says
"use `search_records` to obtain valid ids", not "KeyError". `recovery_skill`
is precisely the probability that the agent acts on that text, and it is the
parameter with the largest effect on the failure rate in this whole
simulation — larger than any topology choice.

The topology numbers say so directly: supervisor and pipeline are slightly
worse than baseline here, because a fresh worker context *discards* the error
history that would tell it not to repeat the call. **Isolation is not free.
It contains cascades and it also amputates learning.**

---

## 4. Context exhaustion

Accumulated observations pass the context window before the task finishes.

**Mechanism.** `context_bloat` inflates a step's observation — a tool that
returns a whole document when asked for a field. The useful content does not
grow; the token count does.

**Reproduction.** `seed=1`, topology `single`, task `t1-audit-ops`,
`context_bloat=0.7`, `context_limit=400`, `max_steps=20`.

```
1. search_records  faults=context_bloat  matched record ids: ... padding padding padding ...
2. fetch_record    faults=-              rec-001: title='Ledger reconciliation' ...
3. fetch_record    faults=context_bloat  rec-004: title='Vendor renewal' ... padding ...
-> context_exhausted: context 529 > limit 400
```

**Rate per topology** (`context_bloat=0.7`, `context_limit=400`):

| `single` | `supervisor` | `pipeline` | `reflexive` |
|---|---|---|---|
| 83.8% | **0.0%** | 83.8% | 10.4% |

**Detection.** A token ceiling on accumulated observations, checked every
step. Distinct from the step cap and it must stay distinct: one bloated
observation costs many tokens in a *single* step, so a step cap does not
bound context and a context cap does not bound loops.

**Mitigation.** **Fresh contexts per unit of work.** This is the one place a
supervisor is structurally, not marginally, better: 0.0% against 83.8% on
identical fault draws, because every worker starts with a clean window and
bloat cannot accumulate across steps. Pipeline scores identically to single
(83.8%) because it *threads context forward* between stages — which is
exactly the design difference, and it shows up as a 84-point gap.

Raising the ceiling is the tempting non-fix: it defers exhaustion by a
constant while bloat grows per step.
`test_context_exhaustion_is_not_fixed_by_a_bigger_window` covers this. The
real fix is upstream — bound what the tool returns.

---

## 5. Partial failure needing rollback

The run fails *after* mutating state, leaving a store that matches neither
the starting state nor the intended one.

**Mechanism.** A multi-step task writes at step k and dies at step k+n. The
notes and deletions from before the failure are still there.

**Reproduction.** `seed=1`, topology `reflexive`, task `t4-note-and-total`,
`bad_args=0.45`, `transient=0.3`, `recovery_skill=0.2`, `max_steps=6`.

```
(first attempt)
1. search_records     faults=-         err=False matched record ids: rec-004
2. fetch_record       faults=bad_args  err=True  -32602: argument 'record_id' must be string
3. fetch_record       faults=transient err=True  transient transport failure: connection reset
4. fetch_record       faults=transient err=True  transient transport failure: connection reset
(reflection, retry)
1. fetch_record       faults=-         err=False rec-004: title='Vendor renewal' ...
2. append_note        faults=bad_args  err=True  -32602: argument 'text' must be string
3. append_note        faults=-         err=False note appended; 1 notes now stored   <-- WRITE
4. summarise_amounts  faults=transient err=True  transient transport failure
5. summarise_amounts  faults=bad_args  err=True  -32602: argument 'record_ids' must be array
6. summarise_amounts  faults=bad_args  err=True  -32602: argument 'record_ids' must be array
-> error_cascade, with 1 journalled write outstanding
```

The run failed at the aggregation step, having already appended a note. The
task did not happen; the note did.

**Detection.** Not an outcome code — the outcome was `error_cascade`. It is
detected by the **journal being non-empty at the point of failure**. Any
terminal state other than success with journal entries outstanding is a
partial failure.

**Mitigation.** **Compensation, plus idempotency declared per tool.**

- `ToolContext.journal` records enough to undo each mutation;
  `ToolContext.rollback()` replays it backwards. Verified by
  `test_partial_failure_leaves_state_that_rollback_can_repair`.
- It is compensation, **not a transaction**. Stdio has no two-phase commit.
  Rollback is best-effort and can itself fail; a doc claiming otherwise would
  be the lie this repo is meant to avoid.
- The complementary control is the `idempotentHint` annotation each tool
  declares. `append_note` is *not* idempotent, so the correct response to the
  `transient` failures above — retry with backoff — would double-write
  without the server's `idempotencyKey` dedupe. **The right retry policy
  corrupts state unless the tool layer knows which calls are replayable.**

---

## 6. Unconfirmed destructive action

An agent deletes something it should have asked about first. Included because
it is the failure whose cost is not measured in tokens.

**Mechanism.** A destructive tool guarded only by a `confirm: bool` flag. A
model that emits `confirm=True` on its first attempt walks straight through
the gate, and models emit plausible-looking arguments constantly.

**Reproduction.** Not a seed — a protocol test, because it must hold for
*every* input rather than at a rate:
`tests/test_protocol_conformance.py::test_a_guessed_token_does_not_open_the_gate`.

```python
delete_record(record_id="rec-005", confirm_token="yes")
-> isError: True, and rec-005 still exists
```

**Rate per topology.** 0% by construction, at every fault rate. That is the
point of a gate as opposed to a mitigation: it does not reduce a probability,
it removes the path.

**Detection.** Not applicable, and that matters. The other five modes are
detected after the fact. This one has to be **prevented**, because there is
no detector that un-deletes a record.

**Mitigation.** **A two-call gate with a server-issued token.** The first
call returns `isError: True` plus a token *and a description of what would be
destroyed*:

```
confirmation required. delete_record would permanently remove rec-005
(title='Access review', owner=sre). To proceed, call again with
confirm_token='5e08221c3aee'.
```

Three properties do the work:

1. **The token must come from the server.** It cannot be guessed or
   hallucinated, which a boolean can.
2. **The gate costs exactly one turn.** Cheap enough to apply to every
   destructive tool without anyone routing around it.
3. **The refusal names the consequence.** A confirmation prompt that does not
   say what it is about to destroy trains its caller to confirm reflexively,
   which is a gate that has been disabled by its own UX.

---

## Summary: which topology helps with which failure

| failure mode | best topology | worst | is topology the answer? |
|---|---|---|---|
| Infinite loop | `supervisor` (0.0%) | `single` (24.6%) | **Yes** — per-unit step budgets |
| Tool misselection | `reflexive` (15.0%) | `pipeline` (44.2%) | **No** — fix the schemas |
| Error cascade | `reflexive` (21.7%) | `pipeline` (56.7%) | **No** — fix the error messages |
| Context exhaustion | `supervisor` (0.0%) | `single`/`pipeline` (83.8%) | **Yes** — fresh contexts |
| Partial failure | — | — | **No** — journal and compensate |
| Unconfirmed destruction | — | — | **No** — gate the tool |

Two of six are genuinely fixed by topology, and both are fixed by the same
underlying property — **isolation of resources per unit of work**, not by
supervision as such. A pipeline gets the step-budget half of that benefit for
a fraction of the supervisor's token cost.

The other four are fixed in the tool layer: schemas, error text, idempotency
declarations, and confirmation gates. None of them care what the graph looks
like.

**The judgement.** Reaching for a multi-agent topology is the expensive
answer to a question that is usually about tool design. The headline result
(`results/topologies.md`) says supervisor buys a real 3.5-point improvement
for 1.73x the tokens while being dominated on both axes by two simpler
designs. This document says why: supervisor's advantage is concentrated in
two failure modes, and a cheaper topology captures most of it. Spend the
effort on the tools first, and add a topology when you can name the specific
failure mode it is buying you.

---

## What this does not establish

- **Nothing about any real model.** The rates are properties of a fault model
  someone chose. A different fault model gives different rates. What
  transfers is the *shape* — which mitigation attacks which mechanism.
- **The fault rates are uncalibrated.** Nothing here shows that a real agent
  misselects tools 55% of the time, or 5%.
- **One task shape.** All six tasks are search-then-fetch-then-aggregate. The
  case where a supervisor is supposed to win — genuinely parallel, separable
  subtasks — is not represented, and the numbers here should not be read as
  evidence against it.
- **`reflexive` gets a retry the others do not.** Part of its advantage is
  simply a second draw from the fault distribution, not reflection.
- **Detection is measured; recovery mostly is not.** Except for the rollback
  path, this document establishes that failures are *caught*, not that a
  system built on these mitigations completes more tasks.

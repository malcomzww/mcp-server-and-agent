# mcp-server-and-agent

A hand-written MCP server at the JSON-RPC level, four agent topologies over
it, and the failure rate of each measured under a controlled fault model.
**The agent brain is a deterministic simulation, not a live LLM** — a
scripted policy executes plans over the real MCP server while a parameterised
fault process perturbs its choices at stated rates. That is the instrument,
not a compromise: a topology's failure rate measured against a live model is
confounded by that model's nondeterminism, and you cannot tell whether a
difference came from the topology or from the sampler. Everything below is a
failure rate of a *topology under a fault model*, never of any real model.

**This repo answers one question:**

> What is each agent topology's failure rate, and does supervisor actually
> beat single-agent?

## The answer

**Supervisor beats single-agent, by less than it costs, and it is dominated
by two simpler designs.**

| topology | failure rate | tokens/run | vs single | steps | handoffs |
|---|---|---|---|---|---|
| `single` | **31.7%** | 2,449 | 1.00x | 5.44 | 0.00 |
| `supervisor` | **28.2%** | 4,238 | 1.73x | 5.16 | 3.88 |
| `pipeline` | **21.2%** | 3,244 | 1.32x | 5.37 | 1.66 |
| `reflexive` | **4.5%** | 2,996 | 1.22x | 6.31 | 0.32 |

600 runs per topology (6 tasks x 100 trials), every topology facing the
identical fault stream for a given (task, trial) — a paired comparison, not
four independent samples.

- Supervisor's 3.5-point gain is **real, not noise**: paired bootstrap delta
  **-0.035 [-0.062, -0.008]** at 95%, interval excludes zero.
- It costs **1.73x the tokens**. `pipeline` and `reflexive` both beat it on
  failure rate **and** on tokens. Paying 1.7x for the third-best outcome is
  the finding.
- The ranking `reflexive < pipeline < supervisor < single` holds across a
  0.5x–2.0x sweep of the fault rates.
- The gaps *narrow* as faults rise. **Topology is a second-order effect; tool
  reliability is the first-order one.**

Full table, confidence intervals and sensitivity sweep:
[`results/topologies.md`](results/topologies.md). Every number is generated
by `scripts/generate_results.py`, which **asserts its own claims and exits
non-zero if they break**.

## The more useful answer

Under the blended fault model above the topologies look similar. Turn one
fault up at a time and they are radically different — and *which* topology
helps depends entirely on *which* fault you have:

| failure mode | best | worst | is topology the answer? |
|---|---|---|---|
| Infinite loop | `supervisor` 0.0% | `single` 24.6% | **Yes** — per-unit step budgets |
| Tool misselection | `reflexive` 15.0% | `pipeline` 44.2% | **No** — fix the schemas |
| Error cascade | `reflexive` 21.7% | `pipeline` 56.7% | **No** — fix the error messages |
| Context exhaustion | `supervisor` 0.0% | `single` 83.8% | **Yes** — fresh contexts |
| Partial failure | — | — | **No** — journal and compensate |
| Unconfirmed destruction | — | — | **No** — gate the tool |

Two of six are fixed by topology, both by the same property — isolation of
resources per unit of work, not supervision as such. The other four are fixed
in the tool layer. Note that supervisor and pipeline are *worse* than single
on error cascades: a fresh worker context discards the error history that
would have told it not to repeat the call. **Isolation contains cascades and
also amputates learning.**

The judgement artifact, with reproduction seeds, real traces and a mitigation
per mode: [`docs/failure-taxonomy.md`](docs/failure-taxonomy.md).

## Scope

Deliberately narrow. In scope: the MCP protocol surface, four topologies, six
failure modes, and the token cost of each. Out of scope: anything that does
not help answer the question above.

## Quickstart

```bash
uv sync --extra dev
uv run pytest -q                          # 90 tests
uv run python scripts/generate_results.py # regenerates results/
uv run python scripts/find_failure_seeds.py
```

Run the MCP server against any client speaking stdio:

```bash
uv run python -m mcp_server_and_agent.server
```

## What is in here

| file | what it is |
|---|---|
| `src/.../protocol.py` | JSON-RPC 2.0 framing, error objects, request validation |
| `src/.../server.py` | MCP lifecycle, dispatch, idempotency dedupe, stdio loop |
| `src/.../tools.py` | 5 tools, 1 resource, 1 prompt, the confirmation gate, rollback journal |
| `src/.../faults.py` | the fault model — the experiment's independent variable |
| `src/.../agent.py` | the scripted policy, the ReAct loop, the task set |
| `src/.../topologies.py` | the four topologies |

### The MCP server is written against the spec, not on an SDK

`initialize` / `tools/list` / `tools/call` / `resources/read` / `prompts/get`
/ `ping`, with correct `-32700` / `-32600` / `-32601` / `-32602` / `-32603`
error objects. The reason is not purity: an SDK hides exactly the seams this
repo measures. Protocol conformance is one of the few things in an agent
stack that is *exactly* testable — a malformed request has one correct error
code — so `tests/test_protocol_conformance.py` covers the cases a happy-path
implementation gets wrong: notifications getting no reply, `"id": null` being
a request rather than a notification, batch rejection, and the boundary
below.

**A tool that does not exist is `-32602`. A tool that exists and fails is a
successful response carrying `isError`.** The first is a bug in the client;
the second is feedback the agent can act on. Collapsing them means the agent
either retries unfixable calls forever or gives up on recoverable ones.

### Four topologies

- **`single`** — one agent, the whole tool list, the whole plan. Baseline.
- **`supervisor`** — a supervisor dispatches each step to a fresh worker.
  Each dispatch pays a handoff tax because the worker starts cold.
- **`pipeline`** — a fixed discover → fetch → aggregate chain. No routing
  decision to get wrong; no re-planner when a stage fails. Context threads
  forward.
- **`reflexive`** — single agent plus exactly one reflection-and-retry pass.
  Chosen as the fourth because it isolates what supervisor confounds:
  **whether a second *attempt* is worth more than a second *agent*.** Both
  cost extra tokens; under this fault model only one adds a capability.

## Provenance

Every number in this README comes from a committed script.

- Date: **2026-08-25**
- Hardware: 24-core CPU, 32 GB RAM, no GPU
- Model: **`simulated-scripted-policy` — no LLM involved**
- Seed: `20260825`
- Reproduce: `python scripts/generate_results.py`
- Raw artifact: `results/topologies-raw.md` (gitignored — per-task detail)
- Committed artifact: `results/topologies.md`

CI regenerates `results/` and fails on `git diff --exit-code`, so a
hand-edited number breaks the build. That gate only means something because
the experiment is deterministic — same seed, same failure rates — which
`tests/test_topologies.py` asserts in both directions.

## Limitations

- **No LLM was involved in any measurement.** The agent brain is a scripted
  policy and the faults are drawn from a distribution I chose. These are
  failure rates of *topologies under a controlled fault model*, **not** of
  any real model in any real deployment. What transfers is the shape of the
  result — which mitigation attacks which mechanism — not the values.
- **The fault rates are uncalibrated.** Nothing here establishes that a real
  agent misselects a tool 16% of the time. Calibrating them requires the live
  runs this repo deliberately does not do, and until someone does that, the
  absolute rates are arbitrary and only the comparisons are meaningful.
- **Token counts are synthetic**: a fixed charge per step plus `len(text)//4`
  for observations. Consistent across topologies so the *ratios* hold; the
  dollar figures forecast nobody's bill.
- **Six tasks, one server, one task shape.** All six are
  search-then-fetch-then-aggregate. The case a supervisor is supposed to win
  — genuinely parallel, separable subtasks — **is not represented here**, and
  these numbers are not evidence against it. This is the single biggest thing
  the repo does not establish.
- **`reflexive` gets one retry the others do not.** Part of its advantage is
  a second draw from the fault distribution rather than reflection as such.
- **No LangGraph.** The brief named it; the agent loop here is hand-written
  in ~200 lines because the failure modes under study are properties of the
  loop, and a framework would have made the step cap, context accounting and
  cascade detector someone else's implementation details.
- **Detection is measured; recovery mostly is not.** Apart from the rollback
  path, this establishes that failures are *caught*, not that a system built
  on these mitigations completes more tasks.

## Built on

- [`llm-client-kit`](https://github.com/malcomzww/llm-client-kit) v0.1.0 —
  `CostLedger` for token and spend accounting.
- [`llm-eval-harness`](https://github.com/malcomzww/llm-eval-harness) v0.1.0 —
  `stats.paired_delta_ci` and `is_reportable` for the confidence intervals,
  `types.RunMeta` for the provenance block.

## License

MIT — see [LICENSE](LICENSE).

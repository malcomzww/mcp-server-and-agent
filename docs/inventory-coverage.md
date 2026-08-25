# Inventory coverage

Anchored to `Bucket_Concept_Inventory.md`, bucket B2 (Agentic & multimodal).

- 2C hand-written MCP server plus LangGraph agent (DO-8)
- 2C ReAct, plan-and-execute, reflection
- 2C topologies and their failure modes
- 2C tool design: schema clarity, granularity, errors as model feedback, idempotency, confirmation gates
- 2C MCP as a protocol: transports, discovery, resources, prompts, sampling, OAuth scope
- 2C state and memory, checkpointing, durable execution
- 2C agent failure modes: loops, tool misselection, error cascades, context exhaustion, partial failure
- 2C reliability: retries with jitter, timeouts, circuit breakers, compensation
- 2C multi-agent handoffs, A2A
- 2C cost and latency control: step caps, per-step routing, parallel tool calls

## Deviations, stated

**DO-8 names a LangGraph agent; this repo hand-writes the loop instead.** The
failure modes under study -- step caps, context accounting, cascade detection
-- are properties of the loop itself, and delegating them to a framework
would have made the thing being measured someone else's implementation
detail. The loop is ~200 lines in `src/mcp_server_and_agent/agent.py`. The
concepts the inventory item covers (ReAct, plan-and-execute, reflection,
handoffs) are all implemented; the named library is not used.

**Sampling and OAuth scope are not implemented.** The MCP protocol item lists
both. Neither is reachable from the one question this repo answers, and
building an unused OAuth path to tick a box is how a narrow repo becomes a
thin one. Transports, discovery, resources and prompts are implemented and
tested.

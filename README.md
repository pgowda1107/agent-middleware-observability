# Agent Middleware Observability Prototype

This workspace contains a runnable prototype for the middleware-based approach
for LangGraph agent observability. It uses:

- `nemo-relay==0.9.0`
- `nemo-relay-cli-bin==0.9.0`
- `langgraph==1.2.11`
- `langchain-nvidia-ai-endpoints==1.4.3`

The demo does not require an API key. It uses deterministic agents and tools so
that trace hierarchy can be validated locally before a live LLM is connected.

## Run

```bash
source .venv/bin/activate
python -m pip install --no-deps -e .
agent-observability-demo
```

Outputs are written to:

- `outputs/trace_events.jsonl`
- `outputs/trace_summary.md`

## What It Builds

The prototype implements this reference architecture:

- a custom Python React-loop orchestrator
- a LangGraph workflow invoked as a tool
- deterministic graph nodes wrapped with Relay middleware
- compiled LangGraph subgraphs for workflow mapping and trace validation subagents
- nested tool scopes under the subagent graph nodes that own each tool call

The runner validates that key parent-child scope edges are present, so a passing
run means the traces are hierarchical rather than flat.

By default the runner uses only the explicit middleware scopes. You can enable
`NemoRelayCallbackHandler` in `ReactLoopOrchestrator(include_langgraph_callbacks=True)`
when you also want LangGraph's built-in runnable scopes, but that intentionally
adds extra framework spans between the workflow tool and each wrapped node.

## Run On An Instance

Docker:

```bash
git clone <your-repo-url>
cd <repo>
docker build -t agent-middleware-observability:latest .
docker run --rm -v "$PWD/outputs:/app/outputs" agent-middleware-observability:latest
```

Plain Python VM:

```bash
git clone <your-repo-url>
cd <repo>
bash scripts/run_demo.sh
```

See `docs/INSTANCE_RUNBOOK.md` for the longer runbook.

## Relay CLI Notes

The matching Relay CLI binary is installed in `.venv/bin/nemo-relay`.

```bash
.venv/bin/nemo-relay --version
.venv/bin/nemo-relay doctor --offline
```

The CLI doctor currently reports no user-level Relay config, which is expected
for this local project setup. It also reports that this desktop Codex bundle has
`codex-cli 0.136.0-alpha.2`, while Relay's Codex CLI integration requires
`0.143.0` or newer. The Python instrumentation used by this project works
without the global Codex CLI integration.

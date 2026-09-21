# Running On An Instance

This project can run on a plain Linux VM or inside Docker. It does not require a
GPU or an API key for the deterministic demo path.

## Option 1: Docker

```bash
git clone <your-repo-url>
cd <repo>
docker build -t agent-middleware-observability:latest .
docker run --rm -v "$PWD/outputs:/app/outputs" agent-middleware-observability:latest
```

The run writes:

- `outputs/trace_events.jsonl`
- `outputs/trace_summary.md`

## Option 2: Python Virtual Environment

```bash
git clone <your-repo-url>
cd <repo>
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/agent-observability-demo
```

Or use:

```bash
bash scripts/run_demo.sh
```

## Health Checks

```bash
.venv/bin/python -m compileall src
.venv/bin/python -m pip check
.venv/bin/nemo-relay doctor --offline
```

## Live Model Hookup

The checked-in demo is deterministic so trace hierarchy is easy to validate. To
connect a live NVIDIA-hosted model later, copy `.env.example` to `.env`, set
`NVIDIA_API_KEY`, and replace the deterministic node logic with a model call
through `langchain-nvidia-ai-endpoints`. Keep the Relay node wrappers in place so
the trace shape remains:

```text
agent:react-orchestrator
  tool:langgraph-observability-workflow
    node:*
      subagent:*
        tool:*
```

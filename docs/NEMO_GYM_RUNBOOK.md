# NeMo Gym Local Runbook

This runbook shows the local NeMo Gym flow for Relay handoff traces. It uses:

- `resources_servers/relay_handoff`: verifier and tool endpoints
- `responses_api_models/scripted_handoff_model`: scripted local Responses API model
- `src/agent_middleware_observability/nemo_gym_pipeline.py`: Relay trace to NeMo Gym JSONL adapter

## Setup

Install the base Relay demo:

```bash
make setup
```

Install NeMo Gym into a separate virtualenv:

```bash
make setup-nemo-gym
```

## Generate Data

Run the managed Relay demo and convert the trace:

```bash
make run-managed
make run-rl-pipeline
make run-model-call-samples
make run-nemo-gym-data
```

`run-nemo-gym-data` writes local data under `outputs/nemo_gym/` and stages
`example.jsonl`, `train.jsonl`, and `validation.jsonl` into the resources server
data directory.

`run-model-call-samples` writes `outputs/model_calls/`. Those rows preserve the
actual model request messages and assistant targets. Use them for SFT/RL data
review; use `outputs/rl/` for structural trajectory and reward records.

## Validate And Run

```bash
make run-nemo-gym-validate
make run-nemo-gym-test
make run-nemo-gym-collate
make run-nemo-gym-rollout
```

The local rollout should produce:

```json
{
  "reward": 1.0,
  "actual_handoff_tools": ["research_agent", "writer_agent"],
  "missing_handoff_tools": [],
  "handoff_order_ok": true
}
```

Outputs are local and ignored by git:

```text
outputs/nemo_gym/rollouts.jsonl
outputs/nemo_gym/rollouts_aggregate_metrics.json
```

## Notebooks

- `notebooks/01_relay_trace_to_rl_dataset.ipynb`
- `notebooks/02_relay_trace_to_nemo_gym_data.ipynb`
- `notebooks/03_nemo_gym_local_rollout.ipynb`

## NeMo RL

This repo validates the NeMo Gym environment locally. Full NeMo RL training is
not run on a local Mac in this setup; use the NVIDIA NeMo RL NGC container on a
GPU instance.

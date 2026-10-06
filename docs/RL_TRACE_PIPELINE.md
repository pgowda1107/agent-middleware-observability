# Sample RL Pipeline For Relay Traces

This project includes a small trace-to-RL data pipeline that converts NeMo Relay
ATOF JSONL events into reward-labeled episodes and transitions.

The pipeline is intentionally lightweight. It does not train a model. It creates
portable datasets that can be used by an eval job, an offline RL trainer, or a
handoff-quality scorer.

## Run

Create a clean trace first:

```bash
.venv/bin/agent-managed-custom-demo
```

Convert the trace:

```bash
.venv/bin/relay-trace-rl-pipeline \
  --input outputs/managed_custom_trace_events.jsonl \
  --output-dir outputs/rl
```

Or use:

```bash
make run-managed
make run-rl-pipeline
```

## Outputs

The converter writes three local files:

- `outputs/rl/episodes.jsonl`
- `outputs/rl/transitions.jsonl`
- `outputs/rl/summary.md`

Generated output files are ignored by git.

These files are structural. They are useful for reward shaping, trace hierarchy
validation, and handoff scoring, but they are not enough to teach a model which
assistant tokens or tool-call arguments to generate.

## Model-Call Samples

Create faithful model-call samples from the same Relay trace:

```bash
.venv/bin/relay-model-call-samples \
  --trace outputs/managed_custom_trace_events.jsonl \
  --output-dir outputs/model_calls
```

The extractor reads each successful Relay `llm` scope and pairs the start event's
request `messages` with the end event's assistant response. It writes:

- `outputs/model_calls/model_call_samples.jsonl`: full sample metadata plus
  request messages and assistant target
- `outputs/model_calls/sft_messages.jsonl`: compact chat/SFT rows with
  `messages = request messages + assistant target`
- `outputs/model_calls/summary.md`: counts and sample inventory

For handoff/tool learning, the important fields are preserved:

- assistant tool calls use `tool_calls[].function.name` and
  `tool_calls[].function.arguments`
- tool results keep the matching `tool_call_id`
- final assistant content is stored as the next target message

Prompt and completion text is preserved so these files should be reviewed before
sharing. Auth-like keys are redacted by default.

## Episode Records

Each line in `episodes.jsonl` contains:

- `episode_id`: Relay propagation root id
- `outcome`: `success`, `error`, or `incomplete`
- `terminal_reward`: episode-level reward
- `metrics`: scope, mark, handoff, error, and depth counts
- `handoffs`: parent-child edges where execution enters a subagent scope
- `trajectory`: ordered scopes with category, status, duration, reward, metadata, and child names

## Transition Records

Each line in `transitions.jsonl` contains one step:

- `state`: active scope, ancestor path, and child count
- `action`: scope name and type, such as `agent`, `tool`, `llm`, or `function`
- `observation`: status, duration, payload shape, marks, and error text if present
- `reward`: step reward, with the terminal reward added to the final step
- `done`: true on the final transition in the episode

## Reward Heuristic

The sample reward function is simple:

- Successful agent, tool, LLM, and function scopes receive small positive rewards.
- Subagent handoff scopes receive an extra positive reward.
- Error scopes receive a negative reward.
- Open scopes receive a negative reward.
- Successful episodes receive a positive terminal reward based on handoff count and trace depth.
- Failed episodes receive a negative terminal reward.

Tune this file for your own reward model. For example, a production scorer might
reward task quality, latency, cost, policy compliance, or handoff correctness.

## Payload Safety

By default, payload values are exported as shapes instead of raw text. That keeps
the dataset focused on trace structure:

```json
{"type": "str", "chars": 57}
```

Use `--include-payloads` only when you explicitly want truncated payload values
in the generated dataset:

```bash
.venv/bin/relay-trace-rl-pipeline \
  --input outputs/managed_custom_trace_events.jsonl \
  --output-dir outputs/rl_with_payloads \
  --include-payloads
```

Secrets and auth-like fields are redacted either way.

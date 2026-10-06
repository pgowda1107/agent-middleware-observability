NEMO_GYM := .venv-nemo-gym/bin/gym
NEMO_GYM_PATH := $(PWD)/.venv-nemo-gym/bin:$(PATH)
NEMO_GYM_CONFIGS := --config resources_servers/relay_handoff/configs/relay_handoff.yaml --config responses_api_models/scripted_handoff_model/configs/scripted_handoff_model.yaml --search-dir .

.PHONY: setup setup-nemo-gym run run-managed run-rl-pipeline run-nemo-gym-data run-nemo-gym-validate run-nemo-gym-test run-nemo-gym-collate run-nemo-gym-rollout check docker-build docker-run relay-doctor

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt
	.venv/bin/python -m pip install --no-deps -e .

setup-nemo-gym:
	python3.13 -m venv .venv-nemo-gym
	.venv-nemo-gym/bin/python -m pip install --upgrade pip setuptools wheel
	.venv-nemo-gym/bin/python -m pip install nemo-gym uv

run:
	.venv/bin/agent-observability-demo

run-managed:
	.venv/bin/agent-managed-custom-demo

run-rl-pipeline:
	.venv/bin/relay-trace-rl-pipeline --input outputs/managed_custom_trace_events.jsonl --output-dir outputs/rl

run-nemo-gym-data:
	.venv/bin/relay-to-nemo-gym-data --trace outputs/managed_custom_trace_events.jsonl --output-dir outputs/nemo_gym
	cp outputs/nemo_gym/example.jsonl resources_servers/relay_handoff/data/example.jsonl
	cp outputs/nemo_gym/train.jsonl resources_servers/relay_handoff/data/train.jsonl
	cp outputs/nemo_gym/validation.jsonl resources_servers/relay_handoff/data/validation.jsonl

run-nemo-gym-validate:
	PATH="$(NEMO_GYM_PATH)" $(NEMO_GYM) env validate $(NEMO_GYM_CONFIGS)

run-nemo-gym-test:
	PATH="$(NEMO_GYM_PATH)" $(NEMO_GYM) env test --resources-server relay_handoff

run-nemo-gym-collate:
	PATH="$(NEMO_GYM_PATH)" $(NEMO_GYM) dataset collate --config resources_servers/relay_handoff/configs/relay_handoff.yaml --resources-server relay_handoff --mode train_preparation --output-dir outputs/nemo_gym/collated

run-nemo-gym-rollout:
	PATH="$(NEMO_GYM_PATH)" $(NEMO_GYM) eval run $(NEMO_GYM_CONFIGS) --agent relay_handoff_simple_agent --input resources_servers/relay_handoff/data/validation.jsonl --output outputs/nemo_gym/rollouts.jsonl --limit 1 --num-repeats 1 --split validation +upload_rollouts_to_wandb=false

check:
	.venv/bin/python -m compileall src
	.venv/bin/python -m pip check

relay-doctor:
	.venv/bin/nemo-relay doctor --offline

docker-build:
	docker build -t agent-middleware-observability:latest .

docker-run:
	docker run --rm -v "$$(pwd)/outputs:/app/outputs" agent-middleware-observability:latest

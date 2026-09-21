.PHONY: setup run check docker-build docker-run relay-doctor

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt
	.venv/bin/python -m pip install --no-deps -e .

run:
	.venv/bin/agent-observability-demo

check:
	.venv/bin/python -m compileall src
	.venv/bin/python -m pip check

relay-doctor:
	.venv/bin/nemo-relay doctor --offline

docker-build:
	docker build -t agent-middleware-observability:latest .

docker-run:
	docker run --rm -v "$$(pwd)/outputs:/app/outputs" agent-middleware-observability:latest

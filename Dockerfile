FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install --no-deps .

RUN mkdir -p /app/outputs

CMD ["agent-observability-demo", "--trace-file", "/app/outputs/trace_events.jsonl", "--summary-file", "/app/outputs/trace_summary.md"]

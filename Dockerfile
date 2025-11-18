# ============================================
#   Air Pollution Project — Dockerfile
#   Python 3.13 + Poetry + ML dependencies
# ============================================

FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir poetry

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY src/ ./src
COPY data/ ./data
COPY output/ ./output
COPY models/ ./models
COPY result/ ./result
COPY README.md ./

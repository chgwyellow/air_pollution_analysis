FROM python:3.13

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libgomp1 \
    git \
    curl \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir poetry

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# COPY src/ ./src
# COPY data/ ./data
# COPY models/ ./models
# COPY output/ ./output
# COPY result/ ./result
# COPY README.md ./

# ============================================
# Air Pollution Dev Container — Dockerfile
# Python 3.13 + Poetry + ML dependencies
# ============================================

FROM python:3.13

# System dependencies for numpy, scipy, sklearn, shap, pandas
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

# Workspace
WORKDIR /app

# Copy dependency files first (for better Docker caching)
COPY ../pyproject.toml ../poetry.lock ./

# Install Poetry
RUN pip install --no-cache-dir poetry

# Install dependencies (no-root = don't install project as a package)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy project files (optional; main use is bind mount)
COPY ../src ./src
COPY ../README.md ./README.md

CMD ["bash"]

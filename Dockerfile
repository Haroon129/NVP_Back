FROM python:3.11-slim

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# -------------------------
# Workdir
# -------------------------
WORKDIR /app

# -------------------------
# Install uv
# -------------------------
RUN python -m pip install --no-cache-dir uv

# -------------------------
# Copy dependency manifests first (cache-friendly)
# -------------------------
COPY pyproject.toml uv.lock ./

# -------------------------
# Environment config
# -------------------------
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV UV_PYTHON=/usr/local/bin/python
ENV UV_PYTHON_DOWNLOADS=never
ENV PYTHONUNBUFFERED=1
ENV PORT=5001

# -------------------------
# Install dependencies
# -------------------------
RUN uv sync --frozen --no-dev

# -------------------------
# Copy app source
# -------------------------
COPY . .

# -------------------------
# Expose port
# -------------------------
EXPOSE 5001

# -------------------------
# Run app
# -------------------------
CMD ["uv", "run", "api.py"]

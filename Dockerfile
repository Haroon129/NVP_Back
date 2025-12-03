FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV UV_PYTHON=/usr/local/bin/python
ENV UV_PYTHON_DOWNLOADS=never
ENV PYTHONUNBUFFERED=1
ENV PORT=5001

RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 5001

CMD ["/app/.venv/bin/python", "api.py"]

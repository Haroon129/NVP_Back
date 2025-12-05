FROM python:3.13-slim

# Dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalamos uv (gestor de dependencias)
RUN python -m pip install --no-cache-dir uv

# Copiamos primero los manifests para aprovechar caché
COPY pyproject.toml uv.lock ./

# Configuración de uv / entorno
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV UV_PYTHON=/usr/local/bin/python
ENV UV_PYTHON_DOWNLOADS=never
ENV PYTHONUNBUFFERED=1
ENV PORT=5001

# Instalamos dependencias según el lockfile (sin dev)
RUN uv sync --frozen --no-dev

# Copiamos el código del proyecto
COPY . .

EXPOSE 5001

# Ejecutamos usando uv run api.py
CMD ["uv", "run", "api.py"]

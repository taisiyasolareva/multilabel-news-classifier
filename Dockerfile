# HuggingFace Spaces Dockerfile
# Spaces will use this file if present (prefers it over Dockerfile.api)
# This is identical to Dockerfile.api but with explicit app.py reference

FROM python:3.10-slim

WORKDIR /app

# System deps: curl for healthchecks, build-essential for any compiled wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements-api.txt requirements-api.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

# Make local packages importable as top-level modules (api/, models/, utils/, etc.)
ENV PYTHONPATH=/app

# Copy code with directory structure intact
COPY api/ api/
COPY utils/ utils/
COPY analysis/ analysis/
COPY monitoring/ monitoring/
COPY models/ models/
COPY config/ config/
COPY scripts/ scripts/

# Copy app.py (HuggingFace Spaces entry point)
COPY app.py app.py

# Note: Model download happens at runtime (in api/main.py startup) via MODEL_URL env var

# Ensure monitoring output directory exists
RUN mkdir -p monitoring/predictions

# HuggingFace Spaces automatically sets PORT env var
# Expose default port (Spaces uses 7860, but PORT env var is set dynamically)
EXPOSE 7860

# Run uvicorn directly, referencing app from app.py
# HuggingFace Spaces sets PORT env var automatically (use shell form to expand env var)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]


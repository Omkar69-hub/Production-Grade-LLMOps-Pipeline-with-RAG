# ──────────────────────────────────────────────────────────────────────────────
#  Stage 1 – dependency builder
#  Installs all Python packages into /install so the runtime stage is clean.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /deps

# System build deps only needed at compile time
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ──────────────────────────────────────────────────────────────────────────────
#  Stage 2 – lean runtime image
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="your-team@example.com"
LABEL description="Production-Grade RAG LLMOps API v2"
LABEL org.opencontainers.image.source="https://github.com/your-org/rag-llmops"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Disable tokeniser parallelism warning
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Runtime system deps (psycopg2 needs libpq, HuggingFace needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Directories required at runtime
RUN mkdir -p /app/vectorstore /app/data /tmp/rag_uploads

# Non-root user for security
RUN groupadd --system appgroup \
 && useradd --system --gid appgroup --no-create-home appuser \
 && chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

# Docker-native health check (uses /health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with 2 uvicorn workers (adjust to CPU count in production)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "warning"]

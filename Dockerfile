# ──────────────────────────────────────────────────────────────────────────────
#  Stage 1 – dependency builder (caches heavy packages separately)
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /deps

COPY requirements.txt .

# Install build tools and compile dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────────────────────────────────────────
#  Stage 2 – lean runtime image
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="you@example.com"
LABEL description="Production-Grade RAG API"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Create directories that will be used at runtime
RUN mkdir -p /app/vectorstore /app/data /tmp/rag_uploads

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

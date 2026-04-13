# syntax=docker/dockerfile:1.4

# =============================================================================
# Stage 1: Builder - Install dependencies with uv
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

# Install uv and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock .python-version ./

# Create virtual environment and install dependencies
# --system: install to system Python (since we're in a container)
# --no-dev: exclude dev dependencies for production (remove for dev builds)
RUN uv sync --system --no-dev --frozen

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

# Install runtime-only dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv for potential runtime package operations
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH" \
    MLFLOW_TRACKING_URI=sqlite:///app/mlflow.db \
    LOG_LEVEL=INFO

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data/processed /app/data/raw /app/models /app/logs /app/mlruns \
    && chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml uv.lock .python-version ./

# Copy MLflow database and configuration (optional - consider mounting in production)
COPY --chown=appuser:appuser mlflow.db ./mlflow.db 2>/dev/null || true
COPY --chown=appuser:appuser .env.example ./.env 2>/dev/null || true

# Copy entrypoint script
COPY --chown=appuser:appentrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Expose port (adjust if your app uses a different port)
EXPOSE 8000

# Health check endpoint (adjust path to match your FastAPI app)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command - override with docker run arguments
CMD ["/app/entrypoint.sh"]
# ⬇️ PASTE DOCKERFILE CONTENT HERE
FROM python:3.11-slim
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY logs/ ./logs/

# Install dependencies
RUN uv sync --frozen --no-cache

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
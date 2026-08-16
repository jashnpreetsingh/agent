# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Fail fast and log immediately: unbuffered output means container logs show
# the agent's progress in real time rather than at exit.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so the layer caches across source edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY fixtures/ ./fixtures/

# Traces are written at runtime; keep them out of the image layers.
RUN mkdir -p traces && \
    useradd --create-home --uid 1000 agent && \
    chown -R agent:agent /app
USER agent

EXPOSE 8501

# CLI by default; arguments are appended:
#   docker run --rm -e GEMINI_API_KEY=... pubmed-agent --query "..."
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]

# For the web UI, override the entrypoint:
#   docker run --rm -p 8501:8501 -e GEMINI_API_KEY=... \
#     --entrypoint streamlit pubmed-agent run src/ui/app.py --server.address 0.0.0.0

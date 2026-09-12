FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# stamp-mcp is on PyPI -- pip resolves it as a normal dependency.
RUN pip install --no-cache-dir ".[http]"

# 0.0.0.0 inside the container; compose exposes it only to the
# terradev-web Caddy network, never to the host's public interface.
ENV ENTROPY_HOST=0.0.0.0 \
    ENTROPY_PORT=8001 \
    ENTROPY_COMMITMENT_LOG=/data/commitments.jsonl

EXPOSE 8001
CMD ["entropy-mcp-http"]

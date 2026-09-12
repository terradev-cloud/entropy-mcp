FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# stamp-mcp is a public sibling repo -- install it from git, then
# entropy-mcp with the http extra (aiohttp).
RUN pip install --no-cache-dir \
    "git+https://github.com/theoddden/Stamp-MCP.git" \
    ".[http]"

# 0.0.0.0 inside the container; compose exposes it only to the
# terradev-web Caddy network, never to the host's public interface.
ENV ENTROPY_HOST=0.0.0.0 \
    ENTROPY_PORT=8001 \
    ENTROPY_COMMITMENT_LOG=/data/commitments.jsonl

EXPOSE 8001
CMD ["entropy-mcp-http"]

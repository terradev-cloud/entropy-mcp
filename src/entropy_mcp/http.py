#!/usr/bin/env python3
"""Streamable-HTTP transport for entropy -- the same synchronous
dispatch() as the stdio server, served over HTTP via aiohttp.

Optional layer: requires `pip install entropy-mcp[http]`. The stdio
core has zero dependencies and never imports this module.

Endpoint doctrine (learned the hard way on stamp-mcp):

- Accept every method on every path that could plausibly be MCP.
  Clients point at whatever URL they were given with whatever method
  their library defaults to. POST /mcp is canonical; POST / is full
  dispatch; POST /sse 307-redirects to /mcp (method and body intact --
  307, not 308, because some clients downgrade POST to GET on 308);
  GET / is service identity; GET /mcp and GET /sse are real SSE
  streams; OPTIONS on every MCP path is a 204 with CORS headers.
- Never let the router reject before the app sees the request. Every
  plausible path is a defined route; everything else hits a clean
  default-deny 404 -- no stack traces, no probe surface.
- JSON-RPC errors travel as HTTP 200 with the error object in the
  body. Clients parse the envelope; an HTTP 4xx reads as a broken
  endpoint, not a bad request.
- /health is exempt from the concurrency semaphore. Rate-limiting
  health checks is how you get pulled from rotation.
- Concurrency is capped by a 100-slot semaphore on the dispatch path:
  fail slow under burst instead of falling over.
- Every request is logged with method, path, status, IP, user-agent.

TLS is NOT done here. Bind to localhost and put a TLS-terminating
reverse proxy (Caddy, nginx) in front.

dispatch() is synchronous and can block for up to ~2s on beacon or
remote-Stamp calls, so handlers run it in an executor rather than on
the event loop.
"""

import asyncio
import json
import os
import sys

from aiohttp import web

from entropy_mcp.server import dispatch, error_response

# Concurrency cap for the dispatch path. /health and static/discovery
# routes are explicitly exempt.
sem = asyncio.Semaphore(100)

_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Accept, Authorization, mcp-session-id",
}

_SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
}


# ---------------------------------------------------------------------------
# Access log: method, path, status, IP, user-agent on every request.
# ---------------------------------------------------------------------------

@web.middleware
async def access_log(request, handler):
    status = 500
    try:
        resp = await handler(request)
        status = resp.status
        return resp
    except web.HTTPException as exc:
        status = exc.status
        raise
    finally:
        ip = request.remote or "-"
        ua = request.headers.get("User-Agent", "-")
        print(f'{ip} "{request.method} {request.path_qs}" {status} "{ua}"',
              file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_response(data, status=200):
    return web.json_response(data, status=status,
                             content_type="application/json",
                             headers=_CORS)


async def _read_json(request):
    try:
        return await request.json()
    except (json.JSONDecodeError, ValueError):
        return None


async def _dispatch_async(message):
    """Run the synchronous dispatch in an executor so beacon/NTP/remote
    Stamp calls never block the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, dispatch, message)


async def _sse_stream(request):
    """A real SSE channel. Entropy never pushes events, but MCP clients
    open this connection and won't consider the server 'connected'
    unless GET returns 200 text/event-stream. Keep-alive comments every
    15 s until disconnect."""
    response = web.StreamResponse(status=200, reason="OK",
                                  headers=_SSE_HEADERS)
    await response.prepare(request)
    try:
        while True:
            await asyncio.sleep(15)
            await response.write(b": keepalive\n\n")
    except (ConnectionResetError, asyncio.CancelledError, Exception):
        pass
    return response


# ---------------------------------------------------------------------------
# MCP endpoints
# ---------------------------------------------------------------------------

async def handle_mcp(request):
    """MCP JSON-RPC 2.0 over HTTP, behind the concurrency semaphore.

    JSON-RPC errors return HTTP 200 with the error object in the body --
    clients parse the envelope, not the status line.
    """
    async with sem:
        body = await _read_json(request)
        if body is None:
            return _json_response(
                error_response(None, -32700, "Parse error"))

        if isinstance(body, list):
            responses = []
            for m in body:
                r = await _dispatch_async(m)
                if r is not None:
                    responses.append(r)
            if not responses:
                return web.Response(status=202)
            return _json_response(responses)

        resp = await _dispatch_async(body)
        if resp is None:
            return web.Response(status=202)
        return _json_response(resp)


async def handle_mcp_get(request):
    """GET on an MCP path opens the SSE channel."""
    return await _sse_stream(request)


async def handle_mcp_options(request):
    """CORS preflight -- 204 with headers on every MCP path."""
    return web.Response(status=204, headers=_CORS)


async def handle_sse_post(request):
    """Legacy SSE clients POST to /sse. 307 preserves method and body --
    308 can downgrade POST to GET in some client libraries."""
    raise web.HTTPTemporaryRedirect("/mcp")


# ---------------------------------------------------------------------------
# Identity, health, discovery
# ---------------------------------------------------------------------------

async def handle_root(request):
    """Service identity for callers that hit the base URL."""
    from entropy_mcp import __version__
    return _json_response({
        "name": "entropy-mcp",
        "version": __version__,
        "description": "Cryptographically sound randomness for agents, "
                       "with verifiable commit-reveal.",
        "endpoints": {
            "mcp": "/mcp",
            "sse": "/sse",
            "health": "/health",
            "agent_card": "/.well-known/agent-card.json",
            "oauth_discovery": "/.well-known/oauth-protected-resource",
        },
    })


async def handle_health(request):
    """Health check -- exempt from the semaphore. Monitors decide
    rotation on this; never rate-limit it."""
    return _json_response({"status": "ok", "server": "entropy"})


def _agent_card():
    from entropy_mcp import __version__
    from entropy_mcp.server import TOOLS
    return {
        "name": "entropy-mcp",
        "version": __version__,
        "description": "Cryptographically sound randomness for agents "
                       "with verifiable commit-reveal proofs.",
        "url": "https://entropy-mcp.terradev.cloud/mcp",
        "capabilities": {"tools": [t["name"] for t in TOOLS]},
        "transport": ["stdio", "streamable-http"],
    }


async def handle_agent_card(request):
    return _json_response(_agent_card())


async def handle_ard(request):
    """Agent Resource Descriptor -- minimal machine-readable identity."""
    return _json_response(_agent_card())


async def handle_ai_catalog(request):
    """ai-catalog.json -- crawler-facing service catalog entry."""
    from entropy_mcp import __version__
    return _json_response({
        "schema": "ai-catalog/v1",
        "services": [{
            "name": "entropy-mcp",
            "version": __version__,
            "type": "mcp-server",
            "endpoint": "https://entropy-mcp.terradev.cloud/mcp",
            "description": "CSPRNG randomness and verifiable "
                           "commit-reveal for agents",
        }],
    })


def _protected_resource_metadata(resource_path="/mcp"):
    """RFC 9728 metadata: public, no authorization servers."""
    return {
        "resource": f"https://entropy-mcp.terradev.cloud{resource_path}",
        "authorization_servers": [],
        "bearer_methods_supported": ["header"],
        "scopes_supported": [],
    }


async def handle_protected_resource_metadata(request):
    return _json_response(_protected_resource_metadata())


async def handle_robots(request):
    return web.Response(
        text="User-agent: *\nAllow: /\n"
             "Sitemap: https://entropy-mcp.terradev.cloud/sitemap.xml\n",
        content_type="text/plain")


async def handle_sitemap(request):
    return web.Response(
        text='<?xml version="1.0" encoding="UTF-8"?>\n'
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
             "  <url><loc>https://entropy-mcp.terradev.cloud/</loc></url>\n"
             "  <url><loc>https://entropy-mcp.terradev.cloud/mcp</loc></url>\n"
             "</urlset>\n",
        content_type="application/xml")


async def handle_not_found(request):
    """Default-deny for everything else. Clean JSON 404 -- scanners hit
    /.env and /wp-admin within hours of going public; give them nothing."""
    return _json_response({"error": "not found"}, status=404)


# ---------------------------------------------------------------------------
# App factory -- the route table is the contract; tests build this
# in-process and check every (method, path) combination.
# ---------------------------------------------------------------------------

def make_app():
    app = web.Application(middlewares=[access_log])

    # Identity + health
    app.router.add_get("/", handle_root)          # HEAD / rides along
    app.router.add_get("/health", handle_health)

    # MCP paths: every plausible method lands somewhere defined.
    app.router.add_post("/", handle_mcp)
    app.router.add_route("OPTIONS", "/", handle_mcp_options)
    app.router.add_get("/mcp", handle_mcp_get)
    app.router.add_get("/mcp/", handle_mcp_get)
    app.router.add_post("/mcp", handle_mcp)
    app.router.add_post("/mcp/", handle_mcp)
    app.router.add_route("OPTIONS", "/mcp", handle_mcp_options)
    app.router.add_route("OPTIONS", "/mcp/", handle_mcp_options)
    app.router.add_get("/sse", handle_mcp_get)
    app.router.add_get("/sse/", handle_mcp_get)
    app.router.add_post("/sse", handle_sse_post)
    app.router.add_post("/sse/", handle_sse_post)
    app.router.add_route("OPTIONS", "/sse", handle_mcp_options)
    app.router.add_route("OPTIONS", "/sse/", handle_mcp_options)

    # Discovery files -- crawlers ask before they index.
    app.router.add_get("/robots.txt", handle_robots)
    app.router.add_get("/sitemap.xml", handle_sitemap)
    app.router.add_get("/ai-catalog.json", handle_ai_catalog)
    app.router.add_get("/.well-known/agent-card.json", handle_agent_card)
    app.router.add_get("/.well-known/agent.json", handle_agent_card)
    app.router.add_get("/.well-known/ard.json", handle_ard)
    app.router.add_get("/.well-known/oauth-protected-resource",
                       handle_protected_resource_metadata)
    app.router.add_get("/.well-known/oauth-protected-resource/",
                       handle_protected_resource_metadata)
    app.router.add_get("/.well-known/oauth-protected-resource/mcp",
                       handle_protected_resource_metadata)
    app.router.add_get("/.well-known/oauth-protected-resource/mcp/",
                       handle_protected_resource_metadata)

    # Default-deny catch-all, registered last so real routes win.
    app.router.add_route("*", "/{tail:.*}", handle_not_found)

    return app


def main():
    host = os.environ.get("ENTROPY_HOST", "127.0.0.1")
    port = int(os.environ.get("ENTROPY_PORT", "8001"))

    print(f"entropy-mcp: HTTP transport on http://{host}:{port}/mcp "
          f"(public)", file=sys.stderr, flush=True)
    print("entropy-mcp: put a TLS-terminating reverse proxy in front",
          file=sys.stderr, flush=True)
    web.run_app(make_app(), host=host, port=port, access_log=None)


if __name__ == "__main__":
    main()

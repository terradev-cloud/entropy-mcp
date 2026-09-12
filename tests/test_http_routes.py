"""Route-table integration tests -- the (method, path) matrix, in-process.

This test class exists because the failure mode it catches is a router
rejecting a request before the app ever sees it. It exercises the route
table, not the handlers: every plausible MCP path must land somewhere
defined, redirects must preserve method, and unknown paths must get a
clean default-deny 404.

Runs in-process via aiohttp's test utilities -- no server, no port.
"""

import json
import unittest

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp.test_utils import AioHTTPTestCase  # noqa: E402

from entropy_mcp import http  # noqa: E402

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
PING = {"jsonrpc": "2.0", "id": 2, "method": "ping"}


class RouteTableTest(AioHTTPTestCase):
    async def get_application(self):
        return http.make_app()

    # -- canonical MCP paths ------------------------------------------------

    async def test_post_mcp(self):
        resp = await self.client.post("/mcp", json=INIT)
        assert resp.status == 200
        body = await resp.json()
        assert body["result"]["serverInfo"]["name"] == "entropy"

    async def test_post_mcp_trailing_slash(self):
        resp = await self.client.post("/mcp/", json=PING)
        assert resp.status == 200
        assert (await resp.json())["id"] == 2

    async def test_post_root_dispatches(self):
        """POST / is full MCP dispatch, not a 405."""
        resp = await self.client.post("/", json=PING)
        assert resp.status == 200
        assert (await resp.json())["result"] == {}

    async def test_get_root_identity(self):
        resp = await self.client.get("/")
        assert resp.status == 200
        body = await resp.json()
        assert body["name"] == "entropy-mcp"
        assert "mcp" in body["endpoints"]

    async def test_head_root(self):
        resp = await self.client.head("/")
        assert resp.status == 200

    async def test_get_mcp_opens_sse(self):
        resp = await self.client.get("/mcp")
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "text/event-stream"
        resp.close()

    async def test_get_sse_opens_stream(self):
        resp = await self.client.get("/sse")
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "text/event-stream"
        resp.close()

    async def test_post_sse_redirects_307(self):
        """307 preserves method+body; following it must reach dispatch."""
        resp = await self.client.post("/sse", json=PING,
                                      allow_redirects=False)
        assert resp.status == 307
        assert resp.headers["Location"] == "/mcp"

    async def test_post_sse_redirect_followed(self):
        resp = await self.client.post("/sse", json=PING,
                                      allow_redirects=True)
        assert resp.status == 200
        assert (await resp.json())["result"] == {}

    async def test_options_everywhere(self):
        for path in ("/", "/mcp", "/mcp/", "/sse", "/sse/"):
            resp = await self.client.options(path)
            assert resp.status == 204, path
            assert "Access-Control-Allow-Origin" in resp.headers

    # -- error semantics ----------------------------------------------------

    async def test_jsonrpc_error_is_http_200(self):
        """Unknown method: 200 with a JSON-RPC error in the body."""
        resp = await self.client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 7, "method": "bogus"})
        assert resp.status == 200
        body = await resp.json()
        assert body["error"]["code"] == -32601

    async def test_parse_error_is_jsonrpc_envelope(self):
        resp = await self.client.post(
            "/mcp", data=b"not json",
            headers={"Content-Type": "application/json"})
        assert resp.status == 200
        body = await resp.json()
        assert body["error"]["code"] == -32700

    async def test_notification_returns_202(self):
        resp = await self.client.post(
            "/mcp", json={"jsonrpc": "2.0",
                          "method": "notifications/initialized"})
        assert resp.status == 202

    # -- health + discovery -------------------------------------------------

    async def test_health(self):
        resp = await self.client.get("/health")
        assert resp.status == 200
        assert (await resp.json())["status"] == "ok"

    async def test_discovery_files(self):
        for path in ("/robots.txt", "/sitemap.xml", "/ai-catalog.json",
                     "/.well-known/agent-card.json",
                     "/.well-known/agent.json",
                     "/.well-known/ard.json",
                     "/.well-known/oauth-protected-resource",
                     "/.well-known/oauth-protected-resource/mcp"):
            resp = await self.client.get(path)
            assert resp.status == 200, path

    # -- default-deny -------------------------------------------------------

    async def test_scanner_paths_404(self):
        for path in ("/.env", "/wp-admin", "/credentials.json",
                     "/admin", "/.git/config"):
            resp = await self.client.get(path)
            assert resp.status == 404, path
            body = await resp.json()
            assert body["error"] == "not found"

    async def test_unknown_post_404(self):
        resp = await self.client.post("/random-path", json={})
        assert resp.status == 404


if __name__ == "__main__":
    unittest.main()

"""Tests for the JSON-RPC dispatch layer and tool registry."""

import json
import sys

import pytest

from entropy_mcp import server
from entropy_mcp import commit as cm


def _call(name, arguments=None, msg_id=1):
    req = {"jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
           "params": {"name": name, "arguments": arguments or {}}}
    resp = server.dispatch(req)
    assert resp["id"] == msg_id
    return resp["result"]


def _payload(result):
    return json.loads(result["content"][0]["text"])


# ---------------------------------------------------------------------------
# Protocol plumbing
# ---------------------------------------------------------------------------

def test_initialize():
    resp = server.dispatch({"jsonrpc": "2.0", "id": 1,
                            "method": "initialize",
                            "params": {"protocolVersion": "2025-03-26"}})
    assert resp["result"]["protocolVersion"] == "2025-03-26"
    assert resp["result"]["serverInfo"]["name"] == "entropy"


def test_initialize_default_version():
    resp = server.dispatch({"jsonrpc": "2.0", "id": 1,
                            "method": "initialize", "params": {}})
    assert resp["result"]["protocolVersion"] == "2024-11-05"


def test_tools_list():
    resp = server.dispatch({"jsonrpc": "2.0", "id": 2,
                            "method": "tools/list"})
    names = [t["name"] for t in resp["result"]["tools"]]
    assert names == ["random", "commitment", "test", "explore"]


def test_notifications_get_no_response():
    assert server.dispatch({"jsonrpc": "2.0",
                            "method": "notifications/initialized"}) is None
    assert server.dispatch({"jsonrpc": "2.0",
                            "method": "tools/list"}) is None  # no id


def test_unknown_method():
    resp = server.dispatch({"jsonrpc": "2.0", "id": 9,
                            "method": "bogus/method"})
    assert resp["error"]["code"] == -32601


def test_ping_and_empty_lists():
    for method, key in (("ping", None), ("resources/list", "resources"),
                        ("prompts/list", "prompts"),
                        ("resources/templates/list", "resourceTemplates")):
        resp = server.dispatch({"jsonrpc": "2.0", "id": 1,
                                "method": method})
        assert resp["result"] == ({key: []} if key else {})


# ---------------------------------------------------------------------------
# Tool calls through the dispatch layer
# ---------------------------------------------------------------------------

def test_call_random_int():
    r = _call("random", {"operation": "int",
                         "params": {"min": 1, "max": 6, "count": 3},
                         "seed": "ab" * 32})
    d = _payload(r)
    assert len(d["values"]) == 3
    assert d["seeded"] is True


def test_call_random_bytes():
    d = _payload(_call("random", {"operation": "bytes",
                                  "params": {"count": 8}}))
    assert len(d["bytes"]) == 16


def test_call_shuffle_choose_sample():
    d = _payload(_call("random", {"operation": "shuffle",
                                  "params": {"items": [1, 2, 3]},
                                  "seed": "cc" * 32}))
    assert sorted(d["shuffled"]) == [1, 2, 3]
    d = _payload(_call("random", {"operation": "choose",
                                  "params": {"items": ["a", "b"],
                                             "k": 1},
                                  "seed": "cc" * 32}))
    assert len(d["items"]) == 1
    d = _payload(_call("random", {"operation": "sample",
                                  "params": {"distribution": "uniform",
                                             "params": {"low": 0,
                                                        "high": 1},
                                             "n": 5},
                                  "seed": "cc" * 32}))
    assert len(d["samples"]) == 5
    assert "params_canonical_hash" in d


def test_call_constrained_explore_testsuite():
    d = _payload(_call("random", {"operation": "constrained",
                                  "params": {"attributes":
                                             {"r": {"x": 1.0}},
                                             "n": 10}}))
    assert len(d["records"]) == 10
    d = _payload(_call("explore", {"arms": ["a", "b"],
                                   "strategy": "ucb1", "state": {}}))
    assert d["arm"] in ("a", "b")
    d = _payload(_call("test", {"sequence": "aa" * 64}))
    assert "tests" in d


def test_call_commit_draw_verify_list():
    d = _payload(_call("commitment", {
        "operation": "draw",
        "draw_spec": {"operation": "random_int",
                      "params": {"min": 1, "max": 10}},
        "description": "via rpc"}))
    assert d["verification"]["valid"] is True
    v = _payload(_call("commitment", {
        "operation": "verify",
        "commitment_record": d["commitment"],
        "reveal_record": d["reveal"]}))
    assert v["valid"] is True
    l = _payload(_call("commitment", {"operation": "list",
                                      "limit": 10}))
    assert l["count"] >= 1


def test_error_shape():
    """Tool errors return isError with the structured error dict."""
    r = _call("random", {"operation": "int",
                         "params": {"min": 10, "max": 1}})
    assert r.get("isError") is True
    err = _payload(r)["error"]
    assert err["code"] == "parameter_error"
    assert err["parameter"] == "min"


def test_bad_operation():
    r = _call("random", {"operation": "bogus"})
    assert r.get("isError") is True
    assert _payload(r)["error"]["parameter"] == "operation"
    r = _call("commitment", {"operation": "bogus"})
    assert r.get("isError") is True


def test_unknown_tool():
    r = _call("no_such_tool")
    assert r.get("isError") is True


# ---------------------------------------------------------------------------
# commit.py error paths and edge cases
# ---------------------------------------------------------------------------

def test_draw_unattested():
    d = _payload(_call("commitment", {
        "operation": "draw",
        "draw_spec": {"operation": "random_int",
                      "params": {"min": 1, "max": 5}},
        "attested": False}))
    assert d["commitment"]["attestation"] is None
    assert d["reveal"]["attestation"] is None
    v = d["verification"]
    assert v["checks"]["commitment_attestation_valid"] is False
    assert v["checks"]["reveal_attestation_valid"] is False


def test_reveal_bad_seed_format():
    c = cm.commit({"operation": "random_int",
                   "params": {"min": 1, "max": 6}}, "x")
    from entropy_mcp.errors import SeedError
    with pytest.raises(SeedError):
        cm.reveal(c["commitment"]["commitment_id"], 12345)


def test_verify_chain_garbage_records():
    v = cm.verify_chain({"not": "a record"}, {"also": "not"})
    assert v["valid"] is False
    assert any("schema" in f for f in v["failures"])


def test_list_commitments_bad_limit():
    with pytest.raises(Exception):
        cm.list_commitments(limit=0)
    with pytest.raises(Exception):
        cm.list_commitments(limit=5000)


def test_commit_with_caller_seed():
    r = cm.commit({"operation": "choose",
                   "params": {"items": ["a", "b", "c"], "k": 1}},
                  "caller-seeded", seed="my-seed-string")
    assert r["seed_generated"] is False
    out = cm.reveal(r["commitment"]["commitment_id"], "my-seed-string")
    assert out["verified"] is True


def test_draw_constrained_and_sample_ops():
    for op, params in (
            ("sample", {"distribution": "exponential",
                        "params": {"rate": 1.0}, "n": 5}),
            ("constrained", {"attributes": {"x": {"a": 1.0}}, "n": 5}),
            ("choose", {"items": [1, 2, 3], "k": 2})):
        d = cm.draw(op, params, description=f"op {op}")
        assert d["verification"]["valid"] is True, op


def test_call_commit_and_reveal_tools():
    c = _payload(_call("commitment", {
        "operation": "commit",
        "draw_spec": {"operation": "random_int",
                      "params": {"min": 1, "max": 6}},
        "description": "via rpc"}))
    cid = c["commitment"]["commitment_id"]
    r = _payload(_call("commitment", {"operation": "reveal",
                                      "commitment_id": cid,
                                      "seed": c["seed"]}))
    assert r["verified"] is True


def test_call_tool_internal_error(monkeypatch):
    """A non-EntropyError inside a tool becomes a structured
    internal_error, not a crash."""
    def boom(name, a):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(server, "_run_tool", boom)
    r = _call("random", {"operation": "int",
                         "params": {"min": 1, "max": 2}})
    assert r.get("isError") is True
    assert _payload(r)["error"]["code"] == "internal_error"


def test_main_stdio_loop(monkeypatch, capsys):
    """Feed main() a session: a parse error, a ping, a notification
    (no response), and a real tools/call."""
    import io
    lines = [
        "not json at all",
        "",
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        json.dumps({"jsonrpc": "2.0",
                    "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "random",
                               "arguments": {"operation": "int",
                                             "params": {"min": 1,
                                                        "max": 2}}}}),
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    server.main()
    out = capsys.readouterr().out.strip().split("\n")
    # three responses: parse error, ping result, tools/call result.
    # The notification gets none.
    assert len(out) == 3
    first = json.loads(out[0])
    assert first["error"]["code"] == -32700
    second = json.loads(out[1])
    assert second["id"] == 1 and second["result"] == {}
    third = json.loads(out[2])
    assert third["id"] == 2
    payload = json.loads(third["result"]["content"][0]["text"])
    assert len(payload["values"]) == 1

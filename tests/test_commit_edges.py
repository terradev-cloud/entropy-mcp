"""Edge-case and failure-path tests for commit.py, beacon.py, and
stamp_client.py remote mode."""

import json
import os

import pytest

from entropy_mcp import beacon, commit as cm, schema, stamp_client
from entropy_mcp.errors import (BeaconUnavailable, CommitmentNotFound,
                                ParameterError, StampUnavailable)

SPEC = {"operation": "random_int",
        "params": {"min": 1, "max": 6, "count": 2}}


# ---------------------------------------------------------------------------
# Commitment log internals
# ---------------------------------------------------------------------------

def test_log_trim(tmp_path, monkeypatch):
    """Log trims to the most recent half when it exceeds the cap."""
    path = tmp_path / "log.jsonl"
    monkeypatch.setenv("ENTROPY_COMMITMENT_LOG", str(path))
    monkeypatch.setattr(cm, "LOG_MAX", 10)
    for i in range(12):
        cm.commit(SPEC, f"c{i}")
    lines = path.read_text().strip().split("\n")
    assert len(lines) <= 10
    assert json.loads(lines[-1])["description"] == "c11"


def test_read_log_tolerates_garbage(tmp_path, monkeypatch):
    path = tmp_path / "log.jsonl"
    path.write_text('{"ok": 1}\nnot json\n\n{"ok": 2}\n')
    monkeypatch.setenv("ENTROPY_COMMITMENT_LOG", str(path))
    assert len(cm._read_log()) == 2


def test_read_log_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROPY_COMMITMENT_LOG",
                       str(tmp_path / "nonexistent" / "x.jsonl"))
    assert cm._read_log() == []


def test_list_commitments_since_filters(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROPY_COMMITMENT_LOG", str(tmp_path / "l.jsonl"))
    cm.commit(SPEC, "first")
    assert cm.list_commitments(since="2000-01-01")["count"] == 1
    assert cm.list_commitments(since="2999-01-01")["count"] == 0


# ---------------------------------------------------------------------------
# commit() validation edges
# ---------------------------------------------------------------------------

def test_commit_description_not_string():
    with pytest.raises(ParameterError):
        cm.commit(SPEC, description=123)


def test_commit_draw_spec_params_not_dict():
    with pytest.raises(ParameterError):
        cm.commit({"operation": "shuffle", "params": [1, 2]})


# ---------------------------------------------------------------------------
# Beacon paths (mocked -- no network in tests)
# ---------------------------------------------------------------------------

def test_commit_with_beacon_anchor(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROPY_COMMITMENT_LOG", str(tmp_path / "l.jsonl"))
    monkeypatch.setattr(beacon, "fetch_latest",
                        lambda: schema.make_beacon_anchor(99, "ab" * 32))
    r = cm.commit(SPEC, "anchored", anchor_beacon=True)
    assert r["commitment"]["beacon_anchor"]["round"] == 99
    assert "beacon_warning" not in r


def test_commit_beacon_unreachable_degrades(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROPY_COMMITMENT_LOG", str(tmp_path / "l.jsonl"))

    def boom():
        raise BeaconUnavailable("down")
    monkeypatch.setattr(beacon, "fetch_latest", boom)
    r = cm.commit(SPEC, "degraded", anchor_beacon=True)
    assert r["commitment"]["beacon_anchor"] is None
    assert "beacon_warning" in r


def test_beacon_unreachable_raises(monkeypatch):
    monkeypatch.setenv("ENTROPY_BEACON_URL", "http://127.0.0.1:1")
    with pytest.raises(BeaconUnavailable):
        beacon.fetch_latest()


def test_verify_anchor_none_and_unreachable(monkeypatch):
    assert beacon.verify_anchor(None) is None
    monkeypatch.setenv("ENTROPY_BEACON_URL", "http://127.0.0.1:1")
    anchor = schema.make_beacon_anchor(1, "ab" * 32)
    assert beacon.verify_anchor(anchor) is None


def test_verify_anchor_match_and_mismatch(monkeypatch):
    anchor = schema.make_beacon_anchor(42, "cd" * 32)
    monkeypatch.setattr(beacon, "fetch_round", lambda r: anchor)
    assert beacon.verify_anchor(anchor) is True
    other = schema.make_beacon_anchor(42, "ef" * 32)
    assert beacon.verify_anchor(other) is False


def test_verify_chain_with_anchor(tmp_path, monkeypatch):
    """Anchored commitment verifies; unreachable beacon -> null check."""
    monkeypatch.setenv("ENTROPY_COMMITMENT_LOG", str(tmp_path / "l.jsonl"))
    anchor = schema.make_beacon_anchor(7, "99" * 32)
    monkeypatch.setattr(beacon, "fetch_latest", lambda: anchor)
    monkeypatch.setattr(beacon, "fetch_round", lambda r: anchor)
    c = cm.commit(SPEC, "anchored", anchor_beacon=True)
    r = cm.reveal(c["commitment"]["commitment_id"], c["seed"])
    v = cm.verify_chain(c["commitment"], r["reveal"])
    assert v["checks"]["beacon_anchor_valid"] is True
    assert v["valid"] is True


def test_verify_chain_anchor_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROPY_COMMITMENT_LOG", str(tmp_path / "l.jsonl"))
    anchor = schema.make_beacon_anchor(7, "99" * 32)
    monkeypatch.setattr(beacon, "fetch_latest", lambda: anchor)
    c = cm.commit(SPEC, "anchored", anchor_beacon=True)
    r = cm.reveal(c["commitment"]["commitment_id"], c["seed"])
    # beacon now returns different randomness for the round
    monkeypatch.setattr(beacon, "fetch_round",
                        lambda n: schema.make_beacon_anchor(7, "00" * 32))
    v = cm.verify_chain(c["commitment"], r["reveal"])
    assert v["checks"]["beacon_anchor_valid"] is False
    assert v["valid"] is False


# ---------------------------------------------------------------------------
# Stamp unavailability: attested paths must fail, not degrade silently
# ---------------------------------------------------------------------------

def test_attest_failure_propagates(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROPY_COMMITMENT_LOG", str(tmp_path / "l.jsonl"))

    def boom(payload):
        raise StampUnavailable("stamp down")
    monkeypatch.setattr(stamp_client, "attest", boom)
    monkeypatch.setattr(cm.stamp, "attest", boom)
    with pytest.raises(StampUnavailable):
        cm.commit(SPEC, "no stamp")


def test_verify_chain_stamp_error_marks_check(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROPY_COMMITMENT_LOG", str(tmp_path / "l.jsonl"))
    c = cm.commit(SPEC, "x")
    r = cm.reveal(c["commitment"]["commitment_id"], c["seed"])

    def boom(att):
        raise StampUnavailable("stamp down")
    monkeypatch.setattr(cm.stamp, "verify", boom)
    v = cm.verify_chain(c["commitment"], r["reveal"])
    assert v["checks"]["commitment_attestation_valid"] is False
    assert v["valid"] is False
    assert any("unverifiable" in f for f in v["failures"])


def test_verify_chain_bad_operation(tmp_path, monkeypatch):
    """A draw_spec naming an unknown op fails outcome_reproducible."""
    monkeypatch.setenv("ENTROPY_COMMITMENT_LOG", str(tmp_path / "l.jsonl"))
    c = cm.commit(SPEC, "x")
    r = cm.reveal(c["commitment"]["commitment_id"], c["seed"])
    import copy
    bad = copy.deepcopy(c["commitment"])
    bad["draw_spec"]["operation"] = "nonexistent_op"
    # keep attestation consistent so only the op check fails
    body = {k: v for k, v in bad.items() if k != "attestation"}
    bad["attestation"] = stamp_client.attest(
        json.loads(json.dumps(body)))
    v = cm.verify_chain(bad, r["reveal"])
    assert v["checks"]["outcome_reproducible"] is False
    assert v["valid"] is False


# ---------------------------------------------------------------------------
# stamp_client remote mode (mocked urllib -- no network)
# ---------------------------------------------------------------------------

def test_remote_call_success(monkeypatch):
    class FakeResp:
        def read(self):
            return json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "result": {"content": [{"type": "text",
                                        "text": '{"uuids": ["u1"]}'}]}
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(stamp_client.urllib.request, "urlopen",
                        lambda req, timeout: FakeResp())
    out = stamp_client._remote_call("http://stamp.example", "generate_uuid",
                                    {"count": 1})
    assert out["uuids"] == ["u1"]


def test_remote_call_unreachable(monkeypatch):
    def boom(req, timeout):
        raise OSError("connection refused")
    monkeypatch.setattr(stamp_client.urllib.request, "urlopen", boom)
    with pytest.raises(StampUnavailable):
        stamp_client._remote_call("http://stamp.example", "get_time", {})


def test_remote_call_rpc_error(monkeypatch):
    class FakeResp:
        def read(self):
            return json.dumps({"jsonrpc": "2.0", "id": 1,
                               "error": {"code": -32601,
                                         "message": "nope"}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(stamp_client.urllib.request, "urlopen",
                        lambda req, timeout: FakeResp())
    with pytest.raises(StampUnavailable):
        stamp_client._remote_call("http://stamp.example", "get_time", {})


def test_jcs_fallback_matches_stamp():
    """The remote-mode JCS fallback must agree with Stamp's canon."""
    import stamp_mcp.canon as canon
    payloads = [
        {"b": 1, "a": 2},
        {"nested": {"z": [1, 2.5, "x"], "a": None}},
        {"unicode": "héllo", "num": 3.14},
        [1, "two", {"three": 3}],
        True, False, None, "str", 42, 2.5,
    ]
    for p in payloads:
        assert stamp_client._jcs(p) == canon.canonicalize(p)


def test_jcs_rejects_nonfinite_and_unknown():
    with pytest.raises(ValueError):
        stamp_client._jcs(float("nan"))
    with pytest.raises(ValueError):
        stamp_client._jcs(float("inf"))
    with pytest.raises(TypeError):
        stamp_client._jcs(object())


def test_available():
    assert stamp_client.available() is True  # stamp_mcp is installed


def test_remote_mode_paths(monkeypatch):
    """Force remote mode and exercise get_time/generate_uuid/attest/
    verify through the mocked _remote_call."""
    monkeypatch.setattr(stamp_client, "_local_stamp",
                        lambda: (None, None))
    calls = []

    def fake_remote(url, tool, arguments, timeout=5.0):
        calls.append(tool)
        if tool == "get_time":
            return {"utc": "2026-09-12T00:00:00+00:00",
                    "server": "time.cloudflare.com",
                    "offset_ms": 1.0, "stratum": 3}
        if tool == "generate_uuid":
            return {"uuids": ["u1", "u2"]}
        if tool == "attest":
            return {"id": "a1", "payload": arguments["payload"],
                    "sha256": "ab" * 32}
        if tool == "verify":
            return {"valid": True}
        raise AssertionError(tool)

    monkeypatch.setattr(stamp_client, "_remote_call", fake_remote)
    t = stamp_client.get_time()
    assert t["utc"].startswith("2026-09-12")
    assert stamp_client.generate_uuid(2) == ["u1", "u2"]
    assert stamp_client.attest({"x": 1})["id"] == "a1"
    assert stamp_client.verify({"id": "a1"})["valid"] is True
    assert calls == ["get_time", "generate_uuid", "attest", "verify"]


def test_remote_mode_canonical_hash_uses_jcs(monkeypatch):
    """With no local stamp_mcp, canonical_hash falls back to local JCS
    -- same standard, same output."""
    import stamp_mcp.canon as canon
    monkeypatch.setattr(stamp_client, "_local_stamp",
                        lambda: (None, None))
    payload = {"b": 1, "a": [2, 3]}
    assert stamp_client.canonical_hash(payload) == \
        canon.canonical_hash(payload)
    assert stamp_client.canonicalize(payload) == canon.canonicalize(payload)


def test_remote_call_tool_error(monkeypatch):
    class FakeResp:
        def read(self):
            return json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "result": {"isError": True,
                           "content": [{"type": "text",
                                        "text": "tool broke"}]}
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(stamp_client.urllib.request, "urlopen",
                        lambda req, timeout: FakeResp())
    with pytest.raises(StampUnavailable):
        stamp_client._remote_call("http://stamp.example", "attest", {})


def test_remote_call_nonjson_text(monkeypatch):
    class FakeResp:
        def read(self):
            return json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "result": {"content": [{"type": "text",
                                        "text": "plain text"}]}
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(stamp_client.urllib.request, "urlopen",
                        lambda req, timeout: FakeResp())
    out = stamp_client._remote_call("http://stamp.example", "x", {})
    assert out == {"text": "plain text"}


# ---------------------------------------------------------------------------
# Beacon malformed responses
# ---------------------------------------------------------------------------

def test_fetch_latest_malformed(monkeypatch):
    monkeypatch.setattr(beacon, "_get", lambda p: {"unexpected": "shape"})
    with pytest.raises(BeaconUnavailable):
        beacon.fetch_latest()


def test_fetch_round_malformed(monkeypatch):
    monkeypatch.setattr(beacon, "_get", lambda p: {"round": "notanint"})
    with pytest.raises(BeaconUnavailable):
        beacon.fetch_round(5)


def test_fetch_latest_and_round_ok(monkeypatch):
    monkeypatch.setattr(
        beacon, "_get",
        lambda p: {"round": 42, "randomness": "ab" * 32})
    a = beacon.fetch_latest()
    assert a["round"] == 42 and a["network"] == "drand-mainnet"
    b = beacon.fetch_round(42)
    assert b["randomness"] == "ab" * 32

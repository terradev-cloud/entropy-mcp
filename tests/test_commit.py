"""Unit tests for commit.py: log behavior, validation, seed hygiene."""

import json
import os
import stat

import pytest

from entropy_mcp import commit as cm
from entropy_mcp import schema
from entropy_mcp.errors import CommitmentNotFound, ParameterError

SPEC = {"operation": "random_int",
        "params": {"min": 1, "max": 6, "count": 3}}


def test_commit_returns_seed_never_stored():
    r = cm.commit(SPEC, "dice")
    rec = r["commitment"]
    assert rec["schema"] == schema.COMMITMENT_SCHEMA
    assert r["seed"] not in json.dumps(rec)  # seed never in the record
    # and never in the log
    with open(cm._log_path()) as f:
        assert r["seed"] not in f.read()


def test_commit_caller_seed_not_echoed():
    seed = "ab" * 32
    r = cm.commit(SPEC, "caller seed", seed=seed)
    assert r["seed"] == seed
    assert r["seed_generated"] is False
    assert seed not in json.dumps(r["commitment"])


def test_commit_log_permissions(tmp_path):
    cm.commit(SPEC, "perm check")
    mode = stat.S_IMODE(os.stat(cm._log_path()).st_mode)
    assert mode == 0o600


def test_commitment_shape():
    r = cm.commit(SPEC, "shape")
    assert schema.check_commitment_shape(r["commitment"]) == []


def test_commit_validates_spec():
    with pytest.raises(ParameterError):
        cm.commit({"operation": "bogus", "params": {}})
    with pytest.raises(ParameterError):
        cm.commit({"operation": "shuffle"})  # params missing
    with pytest.raises(ParameterError):
        cm.commit("not a dict")
    with pytest.raises(ParameterError):
        cm.commit(SPEC, "x" * 600)  # description too long


def test_reveal_unknown_id():
    with pytest.raises(CommitmentNotFound):
        cm.reveal("no-such-id", "ab" * 32)


def test_reveal_seed_mismatch():
    r = cm.commit(SPEC, "mismatch")
    out = cm.reveal(r["commitment"]["commitment_id"], "ff" * 32)
    assert out["verified"] is False
    assert out["reason"] == "seed_mismatch"


def test_list_commitments():
    for i in range(5):
        cm.commit(SPEC, f"c{i}")
    r = cm.list_commitments(limit=3)
    assert r["count"] == 3
    # most recent first
    assert r["commitments"][0]["description"] == "c4"


def test_list_commitments_since():
    cm.commit(SPEC, "old")
    r = cm.list_commitments(since="2999-01-01")
    assert r["count"] == 0


def test_beacon_anchor_null_by_default():
    r = cm.commit(SPEC, "no beacon")
    assert r["commitment"]["beacon_anchor"] is None

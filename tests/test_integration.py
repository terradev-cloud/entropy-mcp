"""Full commit -> draw -> reveal -> verify_chain, plus tamper cases.

Each tamper case asserts the SPECIFIC check that must fail -- a chain
that fails for the wrong reason is not a pass.
"""

import copy

import pytest

from entropy_mcp import commit as cm

SPEC = {"operation": "shuffle",
        "params": {"items": ["alice", "bob", "carol", "dave"]}}


def test_full_chain_valid():
    c = cm.commit(SPEC, "integration")
    cid = c["commitment"]["commitment_id"]
    r = cm.reveal(cid, c["seed"])
    assert r["verified"] is True
    v = r["verification"]
    assert v["valid"] is True
    for check, val in v["checks"].items():
        if check == "beacon_anchor_valid":
            assert val is None  # no anchor requested
        else:
            assert val is True, check


def test_draw_one_call():
    r = cm.draw("random_int", {"min": 1, "max": 100, "count": 5},
                description="one-call")
    assert r["verification"]["valid"] is True
    assert len(r["outcome"]["values"]) == 5
    assert all(1 <= v <= 100 for v in r["outcome"]["values"])


def test_draw_outcome_reproduces():
    """The outcome in the reveal must equal an independent re-execution."""
    r = cm.draw("sample", {"distribution": "normal",
                           "params": {"mean": 0, "sd": 1}, "n": 10})
    outcome = cm.execute_draw("sample",
                              {"distribution": "normal",
                               "params": {"mean": 0, "sd": 1}, "n": 10},
                              r["seed"])
    assert outcome == r["outcome"]


def test_tampered_seed():
    c = cm.commit(SPEC, "tamper seed")
    reveal = cm.reveal(c["commitment"]["commitment_id"], c["seed"])
    bad_reveal = copy.deepcopy(reveal["reveal"])
    bad_reveal["seed"] = "ff" * 32
    v = cm.verify_chain(c["commitment"], bad_reveal)
    assert v["valid"] is False
    assert v["checks"]["seed_produces_commitment_hash"] is False
    assert v["checks"]["outcome_reproducible"] is False


def test_tampered_outcome():
    c = cm.commit(SPEC, "tamper outcome")
    reveal = cm.reveal(c["commitment"]["commitment_id"], c["seed"])
    bad_reveal = copy.deepcopy(reveal["reveal"])
    bad_reveal["outcome"]["shuffled"] = ["dave", "carol", "bob", "alice"]
    v = cm.verify_chain(c["commitment"], bad_reveal)
    assert v["valid"] is False
    assert v["checks"]["outcome_reproducible"] is False
    # the reveal attestation also breaks -- the outcome is inside it
    assert v["checks"]["reveal_attestation_valid"] is False


def test_tampered_params():
    c = cm.commit(SPEC, "tamper params")
    reveal = cm.reveal(c["commitment"]["commitment_id"], c["seed"])
    bad_commit = copy.deepcopy(c["commitment"])
    bad_commit["draw_spec"]["params"]["items"] = ["alice", "bob"]
    v = cm.verify_chain(bad_commit, reveal["reveal"])
    assert v["valid"] is False
    assert v["checks"]["params_hash_matches"] is False
    assert v["checks"]["commitment_attestation_valid"] is False


def test_tampered_attestation():
    c = cm.commit(SPEC, "tamper attestation")
    reveal = cm.reveal(c["commitment"]["commitment_id"], c["seed"])
    bad_commit = copy.deepcopy(c["commitment"])
    bad_commit["attestation"]["sha256"] = "00" * 32
    v = cm.verify_chain(bad_commit, reveal["reveal"])
    assert v["valid"] is False
    assert v["checks"]["commitment_attestation_valid"] is False


def test_verify_chain_no_local_state():
    """verify_chain must work on records not in the local log."""
    c = cm.commit(SPEC, "foreign")
    reveal = cm.reveal(c["commitment"]["commitment_id"], c["seed"])
    # Point the log elsewhere -- verify_chain must still work.
    import os
    os.environ["ENTROPY_COMMITMENT_LOG"] = "/nonexistent/nowhere.jsonl"
    try:
        v = cm.verify_chain(c["commitment"], reveal["reveal"])
        assert v["valid"] is True
    finally:
        del os.environ["ENTROPY_COMMITMENT_LOG"]


def test_constrained_draw_includes_conformance():
    r = cm.draw("constrained",
                {"attributes": {"role": {"eng": 0.6, "pm": 0.4}},
                 "n": 200},
                description="compliance")
    assert r["verification"]["valid"] is True
    assert "conformance" in r["outcome"]
    assert "role" in r["outcome"]["conformance"]

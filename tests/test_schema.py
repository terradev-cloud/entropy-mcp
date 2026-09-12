"""Schema stability: golden record fixtures.

Any change to the emitted record structure must fail these tests loudly.
The fixtures below are frozen literals -- if a code change alters the
shape, update NOTHING here; fix the code or bump the schema version.
"""

import pytest

from entropy_mcp import schema


# ---------------------------------------------------------------------------
# Golden fixtures -- frozen literals, do not edit to match code drift.
# ---------------------------------------------------------------------------

GOLDEN_COMMITMENT = {
    "schema": "entropy.commitment.v1",
    "commitment_id": "3f6b0a52-9e2f-4f3a-9d1c-7a2f0e1b2c3d",
    "seed_hash": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "seed_hash_algo": "sha256",
    "draw_spec": {
        "operation": "shuffle",
        "params": {"items": ["a", "b", "c"]},
        "params_canonical_hash": "ab" * 32,
    },
    "description": "golden fixture",
    "committed_at": {
        "utc": "2026-09-11T00:00:00+00:00",
        "ntp_server": "time.cloudflare.com",
        "offset_ms": 1.25,
        "stratum": 3,
    },
    "beacon_anchor": {
        "network": "drand-mainnet",
        "round": 12345,
        "randomness": "cd" * 32,
    },
    "attestation": {
        "id": "attestation-uuid",
        "attested_at": "2026-09-11T00:00:00+00:00",
        "payload": {},
        "sha256": "ef" * 32,
    },
}

GOLDEN_REVEAL = {
    "schema": "entropy.reveal.v1",
    "commitment_id": "3f6b0a52-9e2f-4f3a-9d1c-7a2f0e1b2c3d",
    "seed": "aa" * 32,
    "seed_verified": True,
    "revealed_at": {
        "utc": "2026-09-11T00:01:00+00:00",
        "ntp_server": "time.cloudflare.com",
        "offset_ms": 1.30,
        "stratum": 3,
    },
    "outcome": {"shuffled": ["c", "a", "b"], "permutation": [2, 0, 1]},
    "outcome_canonical_hash": "12" * 32,
    "reproducible": True,
    "elapsed_seconds": 60.0,
    "attestation": {
        "id": "attestation-uuid-2",
        "attested_at": "2026-09-11T00:01:00+00:00",
        "payload": {},
        "sha256": "34" * 32,
    },
}


# ---------------------------------------------------------------------------
# The builders must emit exactly the golden shapes.
# ---------------------------------------------------------------------------

def test_commitment_builder_matches_golden():
    rec = schema.make_commitment(
        commitment_id=GOLDEN_COMMITMENT["commitment_id"],
        seed_hash=GOLDEN_COMMITMENT["seed_hash"],
        draw_spec=schema.make_draw_spec(
            "shuffle", {"items": ["a", "b", "c"]}, "ab" * 32),
        description="golden fixture",
        committed_at=schema.make_timestamp(
            "2026-09-11T00:00:00+00:00", "time.cloudflare.com", 1.25, 3),
        beacon_anchor=schema.make_beacon_anchor(12345, "cd" * 32),
        attestation=GOLDEN_COMMITMENT["attestation"],
    )
    assert rec == GOLDEN_COMMITMENT


def test_reveal_builder_matches_golden():
    rec = schema.make_reveal(
        commitment_id=GOLDEN_REVEAL["commitment_id"],
        seed_hex="aa" * 32,
        revealed_at=schema.make_timestamp(
            "2026-09-11T00:01:00+00:00", "time.cloudflare.com", 1.30, 3),
        outcome={"shuffled": ["c", "a", "b"], "permutation": [2, 0, 1]},
        outcome_canonical_hash="12" * 32,
        elapsed_seconds=60.0,
        attestation=GOLDEN_REVEAL["attestation"],
    )
    assert rec == GOLDEN_REVEAL


def test_key_sets_are_frozen():
    assert schema.COMMITMENT_KEYS == frozenset({
        "schema", "commitment_id", "seed_hash", "seed_hash_algo",
        "draw_spec", "description", "committed_at", "beacon_anchor",
        "attestation"})
    assert schema.REVEAL_KEYS == frozenset({
        "schema", "commitment_id", "seed", "seed_verified",
        "revealed_at", "outcome", "outcome_canonical_hash",
        "reproducible", "elapsed_seconds", "attestation"})
    assert schema.TIMESTAMP_KEYS == frozenset(
        {"utc", "ntp_server", "offset_ms", "stratum"})
    assert schema.BEACON_ANCHOR_KEYS == frozenset(
        {"network", "round", "randomness"})
    assert schema.DRAW_SPEC_KEYS == frozenset(
        {"operation", "params", "params_canonical_hash"})


def test_shape_checkers_accept_golden():
    assert schema.check_commitment_shape(GOLDEN_COMMITMENT) == []
    assert schema.check_reveal_shape(GOLDEN_REVEAL) == []


def test_shape_checkers_reject_drift():
    bad = dict(GOLDEN_COMMITMENT)
    bad["extra_field"] = True
    assert schema.check_commitment_shape(bad) != []

    bad2 = dict(GOLDEN_COMMITMENT)
    del bad2["seed_hash"]
    assert schema.check_commitment_shape(bad2) != []

    bad3 = dict(GOLDEN_COMMITMENT)
    bad3["schema"] = "entropy.commitment.v2"
    assert schema.check_commitment_shape(bad3) != []

    bad4 = dict(GOLDEN_REVEAL)
    bad4["revealed_at"] = {"utc": "2026-09-11T00:01:00+00:00"}
    assert schema.check_reveal_shape(bad4) != []


def test_nullable_beacon_anchor():
    rec = dict(GOLDEN_COMMITMENT)
    rec["beacon_anchor"] = None
    assert schema.check_commitment_shape(rec) == []

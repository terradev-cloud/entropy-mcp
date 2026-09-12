"""Record schemas for entropy-mcp. Frozen at v1.

Every commitment issued under an unstable schema is a record that may
later need deprecating, so this module is the single source of truth for
the exact shape of both record types. `tests/test_schema.py` holds a
golden fixture for each; any change to the emitted structure fails that
test loudly.

Commitment record v1:

    {
        "schema": "entropy.commitment.v1",
        "commitment_id": "<uuid4>",
        "seed_hash": "<sha256 hex of the seed bytes>",
        "seed_hash_algo": "sha256",
        "draw_spec": {
            "operation": "shuffle | choose | random_int | sample | constrained",
            "params": { ... verbatim caller params ... },
            "params_canonical_hash": "<sha256 of RFC 8785 canonical params>"
        },
        "description": "<caller-supplied, max 512 chars>",
        "committed_at": {
            "utc": "<ISO 8601>",
            "ntp_server": "<server>" | null,
            "offset_ms": <float> | null,
            "stratum": <int> | null
        },
        "beacon_anchor": {
            "network": "drand-mainnet",
            "round": <int>,
            "randomness": "<hex>"
        } | null,
        "attestation": { <stamp attest record> }
    }

    ntp_server / offset_ms / stratum are null when NTP was unreachable and
    the timestamp fell back to the local clock. The keys are always present.

Reveal record v1:

    {
        "schema": "entropy.reveal.v1",
        "commitment_id": "<uuid>",
        "seed": "<hex>",
        "seed_verified": true,
        "revealed_at": { <same shape as committed_at> },
        "outcome": { ... },
        "outcome_canonical_hash": "<hex>",
        "reproducible": true,
        "elapsed_seconds": <float>,
        "attestation": { <stamp attest record> }
    }

Rules:

- `schema` is mandatory and versioned. Any future change increments the
  version; verification logic must dispatch on it.
- `params_canonical_hash` uses RFC 8785 (JCS) canonicalization -- the same
  canonical form Stamp uses -- never raw JSON hashing.
- `params` is stored verbatim. Never return a re-serialized payload.
- `beacon_anchor` is nullable; commitments work without it.
"""

COMMITMENT_SCHEMA = "entropy.commitment.v1"
REVEAL_SCHEMA = "entropy.reveal.v1"

SEED_HASH_ALGO = "sha256"
BEACON_NETWORK = "drand-mainnet"
DESCRIPTION_MAX = 512

VALID_OPERATIONS = ("shuffle", "choose", "random_int", "sample", "constrained")

# Exact key sets, used by the schema test to catch accidental drift.
COMMITMENT_KEYS = frozenset({
    "schema", "commitment_id", "seed_hash", "seed_hash_algo", "draw_spec",
    "description", "committed_at", "beacon_anchor", "attestation",
})
DRAW_SPEC_KEYS = frozenset({"operation", "params", "params_canonical_hash"})
TIMESTAMP_KEYS = frozenset({"utc", "ntp_server", "offset_ms", "stratum"})
BEACON_ANCHOR_KEYS = frozenset({"network", "round", "randomness"})
REVEAL_KEYS = frozenset({
    "schema", "commitment_id", "seed", "seed_verified", "revealed_at",
    "outcome", "outcome_canonical_hash", "reproducible",
    "elapsed_seconds", "attestation",
})


def make_timestamp(utc, ntp_server=None, offset_ms=None, stratum=None):
    """Assemble a timestamp block in the frozen shape.

    ntp_server/offset_ms/stratum are null when the time came from the
    local clock rather than an NTP query.
    """
    return {
        "utc": utc,
        "ntp_server": ntp_server,
        "offset_ms": offset_ms,
        "stratum": stratum,
    }


def make_beacon_anchor(round_number, randomness_hex):
    """Assemble a beacon anchor block in the frozen shape."""
    return {
        "network": BEACON_NETWORK,
        "round": round_number,
        "randomness": randomness_hex,
    }


def make_draw_spec(operation, params, params_canonical_hash):
    """Assemble a draw_spec block. `params` is stored verbatim."""
    return {
        "operation": operation,
        "params": params,
        "params_canonical_hash": params_canonical_hash,
    }


def make_commitment(commitment_id, seed_hash, draw_spec, description,
                    committed_at, beacon_anchor, attestation):
    """Assemble a commitment record v1 in the frozen shape."""
    return {
        "schema": COMMITMENT_SCHEMA,
        "commitment_id": commitment_id,
        "seed_hash": seed_hash,
        "seed_hash_algo": SEED_HASH_ALGO,
        "draw_spec": draw_spec,
        "description": description,
        "committed_at": committed_at,
        "beacon_anchor": beacon_anchor,
        "attestation": attestation,
    }


def make_reveal(commitment_id, seed_hex, revealed_at, outcome,
                outcome_canonical_hash, elapsed_seconds, attestation,
                seed_verified=True, reproducible=True):
    """Assemble a reveal record v1 in the frozen shape."""
    return {
        "schema": REVEAL_SCHEMA,
        "commitment_id": commitment_id,
        "seed": seed_hex,
        "seed_verified": seed_verified,
        "revealed_at": revealed_at,
        "outcome": outcome,
        "outcome_canonical_hash": outcome_canonical_hash,
        "reproducible": reproducible,
        "elapsed_seconds": elapsed_seconds,
        "attestation": attestation,
    }


def check_commitment_shape(record):
    """Return a list of structural problems with a commitment record.

    Empty list means the record matches the v1 shape. This checks keys
    and types only -- it does not verify hashes or attestations.
    """
    problems = []
    if not isinstance(record, dict):
        return ["record is not an object"]
    if record.get("schema") != COMMITMENT_SCHEMA:
        problems.append(f"schema is {record.get('schema')!r}, "
                        f"expected {COMMITMENT_SCHEMA!r}")
    extra = set(record) - COMMITMENT_KEYS
    missing = COMMITMENT_KEYS - set(record)
    if extra:
        problems.append(f"unexpected keys: {sorted(extra)}")
    if missing:
        problems.append(f"missing keys: {sorted(missing)}")
    spec = record.get("draw_spec")
    if isinstance(spec, dict):
        if set(spec) != DRAW_SPEC_KEYS:
            problems.append(
                f"draw_spec keys {sorted(spec)} != {sorted(DRAW_SPEC_KEYS)}")
        elif spec.get("operation") not in VALID_OPERATIONS:
            problems.append(f"unknown operation {spec.get('operation')!r}")
    ts = record.get("committed_at")
    if isinstance(ts, dict) and set(ts) != TIMESTAMP_KEYS:
        problems.append("committed_at keys do not match timestamp shape")
    anchor = record.get("beacon_anchor")
    if anchor is not None and (
            not isinstance(anchor, dict)
            or set(anchor) != BEACON_ANCHOR_KEYS):
        problems.append("beacon_anchor keys do not match anchor shape")
    return problems


def check_reveal_shape(record):
    """Return a list of structural problems with a reveal record."""
    problems = []
    if not isinstance(record, dict):
        return ["record is not an object"]
    if record.get("schema") != REVEAL_SCHEMA:
        problems.append(f"schema is {record.get('schema')!r}, "
                        f"expected {REVEAL_SCHEMA!r}")
    extra = set(record) - REVEAL_KEYS
    missing = REVEAL_KEYS - set(record)
    if extra:
        problems.append(f"unexpected keys: {sorted(extra)}")
    if missing:
        problems.append(f"missing keys: {sorted(missing)}")
    ts = record.get("revealed_at")
    if isinstance(ts, dict) and set(ts) != TIMESTAMP_KEYS:
        problems.append("revealed_at keys do not match timestamp shape")
    return problems

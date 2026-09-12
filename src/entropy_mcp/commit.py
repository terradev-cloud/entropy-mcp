"""Tier 1 -- commitment / reveal.

The trust proposition in four commands:

    commit        -- hash a seed, bind it to a draw spec, attest the
                     record via Stamp. The seed goes to the caller and
                     is NEVER persisted server-side.
    reveal        -- caller returns the seed; we verify it hashes to the
                     committed seed_hash, re-execute the draw
                     deterministically, and attest the reveal record.
    draw          -- commit + execute + reveal in one call. The fair
                     path is the easy path.
    verify_chain  -- pure verification of a commitment + reveal pair
                     from ANY source. No local state. This is what a
                     third party runs.

Commitment log: append-only JSONL at ~/.entropy/commitments.jsonl
(override: ENTROPY_COMMITMENT_LOG), mode 0600, capped at 10,000
records. Seeds are never written to it.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

from entropy_mcp import schema
from entropy_mcp import beacon as _beacon
from entropy_mcp import stamp_client as stamp
from entropy_mcp.errors import (
    CommitmentNotFound, ParameterError, SeedError, StampUnavailable,
)

COMMITMENT_LOG = os.environ.get(
    "ENTROPY_COMMITMENT_LOG",
    os.path.expanduser("~/.entropy/commitments.jsonl"))
LOG_MAX = 10000  # trim to most recent half when exceeded (Stamp convention)


def _log_path():
    """Re-read the env var each call so tests can redirect the log."""
    return os.environ.get("ENTROPY_COMMITMENT_LOG", COMMITMENT_LOG)


# ---------------------------------------------------------------------------
# Commitment log: append-only JSONL, 0600, capped, never stores seeds.
# ---------------------------------------------------------------------------

def _append_log(record):
    path = _log_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    _trim_log(path)


def _trim_log(path):
    try:
        with open(path) as f:
            lines = f.readlines()
        if len(lines) > LOG_MAX:
            with open(path, "w") as f:
                f.writelines(lines[LOG_MAX // 2:])
    except OSError:
        pass


def _read_log():
    path = _log_path()
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return records


def _find_commitment(commitment_id):
    for rec in _read_log():
        if rec.get("commitment_id") == commitment_id:
            return rec
    raise CommitmentNotFound(
        f"no commitment with id {commitment_id!r} in the local log",
        parameter="commitment_id",
        expected="a commitment_id returned by commit or draw")


def list_commitments(limit=50, since=None):
    """Read-only retrieval from the commitment log.

    limit: max records returned (most recent first in the response,
    though stored oldest-first). since: optional ISO 8601 string; only
    commitments at or after it are returned.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) \
            or not 1 <= limit <= 1000:
        raise ParameterError("limit must be an integer between 1 and 1000",
                             parameter="limit", expected="1..1000")
    records = _read_log()
    if since is not None:
        records = [r for r in records
                   if (r.get("committed_at") or {}).get("utc", "") >= since]
    records = records[-limit:]
    records.reverse()
    return {"count": len(records), "commitments": records,
            "log": _log_path()}


# ---------------------------------------------------------------------------
# Draw execution: re-runnable, deterministic given (operation, params, seed).
# ---------------------------------------------------------------------------

def execute_draw(operation, params, seed_hex):
    """Execute a draw spec against a seed. Pure -- no log, no Stamp.

    Returns the outcome object that goes into the reveal record. Must
    produce byte-identical output for the same inputs, forever -- the
    seeded stream contract in source.py is what makes that true.
    """
    if operation == "random_int":
        r = _source().random_int(params["min"], params["max"],
                                 params.get("count", 1), seed=seed_hex)
        return {"values": r["values"]}
    if operation == "shuffle":
        r = _source().shuffle(params["items"], seed=seed_hex)
        return {"shuffled": r["shuffled"], "permutation": r["permutation"]}
    if operation == "choose":
        r = _source().choose(
            params["items"], k=params.get("k", 1),
            weights=params.get("weights"),
            replacement=params.get("replacement", False), seed=seed_hex)
        return {"items": r["items"], "indices": r["indices"],
                "probabilities": r["probabilities"]}
    if operation == "sample":
        from entropy_mcp import distributions
        r = distributions.sample(params["distribution"],
                                 params.get("params", {}),
                                 n=params.get("n", 1), seed=seed_hex)
        return {"samples": r["samples"], "summary": r["summary"]}
    if operation == "constrained":
        from entropy_mcp import constrained
        r = constrained.sample_constrained(params["attributes"],
                                           params["n"], seed=seed_hex)
        return {"records": r["records"], "conformance": r["conformance"]}
    raise ParameterError(
        f"unknown operation {operation!r}",
        parameter="operation",
        expected=" | ".join(schema.VALID_OPERATIONS))


def _source():
    from entropy_mcp import source
    return source


def _validate_draw_spec(draw_spec):
    if not isinstance(draw_spec, dict):
        raise ParameterError("draw_spec must be an object",
                             parameter="draw_spec", expected="object")
    op = draw_spec.get("operation")
    if op not in schema.VALID_OPERATIONS:
        raise ParameterError(
            f"draw_spec.operation must be one of "
            f"{list(schema.VALID_OPERATIONS)}, got {op!r}",
            parameter="operation",
            expected=" | ".join(schema.VALID_OPERATIONS))
    params = draw_spec.get("params")
    if not isinstance(params, dict):
        raise ParameterError("draw_spec.params must be an object",
                             parameter="params", expected="object")
    return op, params


def _parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# commit / reveal / draw
# ---------------------------------------------------------------------------

def commit(draw_spec, description="", seed=None, anchor_beacon=False):
    """Create a commitment record. The seed goes to the caller only.

    If seed is None, 32 bytes are generated from os.urandom and returned
    in the response. The seed is never written to the log or stored in
    the record -- only its sha256.
    """
    op, params = _validate_draw_spec(draw_spec)
    if not isinstance(description, str):
        raise ParameterError("description must be a string",
                             parameter="description", expected="string")
    if len(description) > schema.DESCRIPTION_MAX:
        raise ParameterError(
            f"description exceeds {schema.DESCRIPTION_MAX} chars",
            parameter="description",
            expected=f"<= {schema.DESCRIPTION_MAX} chars")

    if seed is None:
        seed_bytes = os.urandom(32)
        seed_hex = seed_bytes.hex()
        seed_generated = True
    else:
        from entropy_mcp.source import normalize_seed
        seed_bytes, seed_hex = normalize_seed(seed)
        seed_generated = False
    seed_hash = hashlib.sha256(seed_bytes).hexdigest()

    params_hash = stamp.canonical_hash(params)
    t = stamp.get_time()
    committed_at = schema.make_timestamp(
        t["utc"], t["ntp_server"], t["offset_ms"], t["stratum"])

    beacon_anchor = None
    beacon_warning = None
    if anchor_beacon:
        try:
            beacon_anchor = _beacon.fetch_latest()
        except Exception as exc:
            beacon_warning = f"beacon unreachable, anchor omitted: {exc}"

    commitment_id = stamp.generate_uuid(1)[0]
    record = schema.make_commitment(
        commitment_id=commitment_id,
        seed_hash=seed_hash,
        draw_spec=schema.make_draw_spec(op, params, params_hash),
        description=description,
        committed_at=committed_at,
        beacon_anchor=beacon_anchor,
        attestation=None,
    )
    # Attest a deep copy WITHOUT the attestation key -- embedding the
    # record itself would create a circular object, and a shallow copy
    # would let later mutations alias into the attested payload.
    body = json.loads(json.dumps(
        {k: v for k, v in record.items() if k != "attestation"}))
    try:
        record["attestation"] = stamp.attest(body)
    except StampUnavailable:
        raise
    except Exception as exc:
        raise StampUnavailable(f"attestation failed: {exc}") from exc

    _append_log(record)

    out = {"commitment": record, "seed": seed_hex,
           "seed_generated": seed_generated}
    if beacon_warning:
        out["beacon_warning"] = beacon_warning
    return out


def _build_reveal(commitment, seed_hex, seed_bytes):
    """Execute the draw and assemble + attest the reveal record."""
    spec = commitment["draw_spec"]
    outcome = execute_draw(spec["operation"], spec["params"], seed_hex)
    outcome_hash = stamp.canonical_hash(outcome)

    t = stamp.get_time()
    revealed_at = schema.make_timestamp(
        t["utc"], t["ntp_server"], t["offset_ms"], t["stratum"])
    try:
        elapsed = (_parse_iso(revealed_at["utc"])
                   - _parse_iso(commitment["committed_at"]["utc"])
                   ).total_seconds()
    except Exception:
        elapsed = None

    record = schema.make_reveal(
        commitment_id=commitment["commitment_id"],
        seed_hex=seed_hex,
        revealed_at=revealed_at,
        outcome=outcome,
        outcome_canonical_hash=outcome_hash,
        elapsed_seconds=elapsed,
        attestation=None,
        seed_verified=True,
        reproducible=True,
    )
    body = json.loads(json.dumps(
        {k: v for k, v in record.items() if k != "attestation"}))
    try:
        record["attestation"] = stamp.attest(body)
    except StampUnavailable:
        raise
    except Exception as exc:
        raise StampUnavailable(f"attestation failed: {exc}") from exc
    return record


def reveal(commitment_id, seed):
    """Reveal a committed draw. Verifies the seed, re-executes the draw,
    attests the reveal record, and returns the full verification block."""
    commitment = _find_commitment(commitment_id)

    from entropy_mcp.source import normalize_seed
    seed_bytes, seed_hex = normalize_seed(seed)
    if hashlib.sha256(seed_bytes).hexdigest() != commitment["seed_hash"]:
        return {
            "verified": False,
            "reason": "seed_mismatch",
            "commitment_id": commitment_id,
        }

    reveal_record = _build_reveal(commitment, seed_hex, seed_bytes)
    verification = verify_chain(commitment, reveal_record)
    return {
        "verified": verification["valid"],
        "commitment": commitment,
        "reveal": reveal_record,
        "verification": verification,
    }


def draw(operation, params, description="", attested=True,
         anchor_beacon=False, seed=None):
    """Commit + execute + reveal in one call. The fair path, made easy.

    With attested=True (the default) Stamp is required; without it the
    records carry attestation: null and verification will report the
    attestations as missing -- useful for local testing only.
    """
    result = commit({"operation": operation, "params": params},
                    description=description, seed=seed,
                    anchor_beacon=anchor_beacon)
    commitment = result["commitment"]
    seed_hex = result["seed"]

    if not attested:
        # Strip the attestation the caller asked us to skip. The record
        # shape is preserved; attestation is null and marked.
        commitment["attestation"] = None

    from entropy_mcp.source import normalize_seed
    seed_bytes, _ = normalize_seed(seed_hex)
    reveal_record = _build_reveal(commitment, seed_hex, seed_bytes)
    if not attested:
        reveal_record["attestation"] = None

    verification = verify_chain(commitment, reveal_record)
    return {
        "commitment": commitment,
        "reveal": reveal_record,
        "outcome": reveal_record["outcome"],
        "seed": seed_hex,
        "verification": verification,
    }


# ---------------------------------------------------------------------------
# verify_chain: pure verification, no local state.
# ---------------------------------------------------------------------------

def verify_chain(commitment_record, reveal_record):
    """Verify a commitment + reveal pair from any source.

    No dependency on the local commitment log -- a third party can
    verify records issued by another Entropy instance. Returns a verdict
    block with each check as a named boolean and a failures list.
    """
    checks = {
        "seed_produces_commitment_hash": False,
        "commitment_attestation_valid": False,
        "reveal_attestation_valid": False,
        "commitment_precedes_reveal": False,
        "outcome_reproducible": False,
        "params_hash_matches": False,
        "beacon_anchor_valid": None,
    }
    failures = []

    shape_problems = schema.check_commitment_shape(commitment_record)
    shape_problems += schema.check_reveal_shape(reveal_record)
    if shape_problems:
        failures.extend(f"schema: {p}" for p in shape_problems)
        return {"valid": False, "checks": checks, "failures": failures}

    # 1. The revealed seed hashes to the committed seed_hash.
    try:
        from entropy_mcp.source import normalize_seed
        seed_bytes, _ = normalize_seed(reveal_record["seed"])
        ok = (hashlib.sha256(seed_bytes).hexdigest()
              == commitment_record["seed_hash"])
        checks["seed_produces_commitment_hash"] = ok
        if not ok:
            failures.append("revealed seed does not hash to seed_hash")
    except Exception as exc:
        failures.append(f"seed check error: {exc}")

    # 2/3. Stamp attestations on both records. Two parts per check:
    # the attestation's own hash must verify, AND the attested payload
    # must equal this record minus its attestation field -- otherwise a
    # tamperer could edit a top-level field while the attestation on the
    # original payload still verifies.
    for key, rec, label in (
            ("commitment_attestation_valid", commitment_record, "commitment"),
            ("reveal_attestation_valid", reveal_record, "reveal")):
        att = rec.get("attestation")
        if not isinstance(att, dict):
            failures.append(f"{label} attestation missing")
            continue
        try:
            res = stamp.verify(att)
            ok = bool(res.get("valid"))
            if not ok:
                failures.append(f"{label} attestation invalid: "
                                f"{res.get('reason', 'hash mismatch')}")
            else:
                body = {k: v for k, v in rec.items() if k != "attestation"}
                if stamp.canonical_hash(att.get("payload")) \
                        != stamp.canonical_hash(body):
                    ok = False
                    failures.append(f"{label} attestation payload does "
                                    f"not match the record body")
            checks[key] = ok
        except Exception as exc:
            failures.append(f"{label} attestation unverifiable: {exc}")

    # 4. Commit timestamp precedes reveal timestamp.
    try:
        c = _parse_iso(commitment_record["committed_at"]["utc"])
        r = _parse_iso(reveal_record["revealed_at"]["utc"])
        checks["commitment_precedes_reveal"] = c <= r
        if not checks["commitment_precedes_reveal"]:
            failures.append("reveal timestamp precedes commit timestamp")
    except Exception as exc:
        failures.append(f"timestamp comparison failed: {exc}")

    # 5. The draw re-executes to the revealed outcome, AND the presented
    # outcome field must hash to outcome_canonical_hash -- checking only
    # the recomputed draw would miss a tampered outcome payload.
    try:
        spec = commitment_record["draw_spec"]
        outcome = execute_draw(spec["operation"], spec["params"],
                               reveal_record["seed"])
        recomputed = stamp.canonical_hash(outcome)
        presented = stamp.canonical_hash(reveal_record["outcome"])
        ok = (recomputed == reveal_record["outcome_canonical_hash"]
              and presented == reveal_record["outcome_canonical_hash"])
        checks["outcome_reproducible"] = ok
        if not ok:
            failures.append("re-executed draw does not match the "
                            "revealed outcome")
    except Exception as exc:
        failures.append(f"outcome reproduction failed: {exc}")

    # 6. The stored params hash matches the params actually committed.
    try:
        spec = commitment_record["draw_spec"]
        ok = (stamp.canonical_hash(spec["params"])
              == spec["params_canonical_hash"])
        checks["params_hash_matches"] = ok
        if not ok:
            failures.append("draw_spec.params does not hash to "
                            "params_canonical_hash")
    except Exception as exc:
        failures.append(f"params hash check failed: {exc}")

    # 7. Beacon anchor (nullable check).
    anchor = commitment_record.get("beacon_anchor")
    if anchor is not None:
        res = _beacon.verify_anchor(anchor)
        checks["beacon_anchor_valid"] = res
        if res is False:
            failures.append("beacon anchor randomness does not match "
                            "the referenced drand round")
        elif res is None:
            failures.append("beacon anchor present but drand "
                            "unreachable -- unverified")

    valid = all(v for v in checks.values() if v is not None)
    return {"valid": valid, "checks": checks, "failures": failures}
    return {"valid": valid, "checks": checks, "failures": failures}

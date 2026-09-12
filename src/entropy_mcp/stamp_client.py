"""Thin client to stamp-mcp for time, attestation, canonicalization, UUIDs.

Two modes:

- Local: import stamp_mcp directly. Preferred -- no network for
  canonical_hash/verify, and attest/get_time degrade to the local clock
  when NTP is unreachable (Stamp's own behavior, preserved here).
- Remote: JSON-RPC tools/call over HTTP POST to a Stamp endpoint,
  default https://stamp-mcp.terradev.cloud, configurable via
  ENTROPY_STAMP_URL. Uses urllib so the zero-dependency core stays
  zero-dependency.

Degradation policy: if Stamp is unavailable in BOTH modes, commit,
reveal, and draw (with attested=True) fail with StampUnavailable rather
than silently producing an unattested record. An unattested commitment
looks identical to an attested one and is worthless. Tier 0 and tier 2
commands never touch this module.

canonical_hash in remote mode: Stamp's HTTP API does not expose
canonical_hash as a tool, so remote mode falls back to a local RFC 8785
(JCS) implementation -- the same public standard Stamp's canon module
implements, so the hashes interoperate. Local mode always uses Stamp's
own canonical_hash.
"""

import hashlib
import json
import os
import urllib.request
import uuid as _uuid
from datetime import datetime, timezone

from entropy_mcp.errors import StampUnavailable

DEFAULT_STAMP_URL = "https://stamp-mcp.terradev.cloud"
REMOTE_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Local mode: stamp_mcp imported lazily so tier 0/2 work without it.
# ---------------------------------------------------------------------------

def _local_stamp():
    try:
        import stamp_mcp.server as srv
        import stamp_mcp.canon as canon
        return srv, canon
    except ImportError:
        return None, None


# ---------------------------------------------------------------------------
# Remote mode: raw JSON-RPC over urllib. No aiohttp in the core.
# ---------------------------------------------------------------------------

def _remote_call(url, tool, arguments, timeout=REMOTE_TIMEOUT):
    """One tools/call against a remote Stamp endpoint. Returns the parsed
    tool result payload, or raises StampUnavailable."""
    req_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    endpoint = url.rstrip("/") + "/mcp"
    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(req_body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Accept": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise StampUnavailable(
            f"remote Stamp at {endpoint} unreachable: {exc}",
            parameter=None, expected="reachable stamp-mcp endpoint") from exc
    if "error" in body:
        raise StampUnavailable(
            f"remote Stamp error: {body['error']}",
            expected="successful tools/call")
    result = body.get("result") or {}
    if result.get("isError"):
        text = "".join(c.get("text", "")
                       for c in result.get("content", []))
        raise StampUnavailable(f"remote Stamp tool error: {text}")
    text = "".join(c.get("text", "") for c in result.get("content", []))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


# ---------------------------------------------------------------------------
# RFC 8785 (JCS) fallback for remote-mode canonical_hash only.
# Interoperable with stamp_mcp.canon -- same public standard.
# ---------------------------------------------------------------------------

def _jcs(obj):
    """Minimal RFC 8785 canonicalization. Remote-mode fallback only."""
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, int) and not isinstance(obj, bool):
        return _jcs(float(obj))
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError("non-finite number is not valid JSON")
        if obj == int(obj) and abs(obj) < 1e21:
            return str(int(obj))
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_jcs(i) for i in obj) + "]"
    if isinstance(obj, dict):
        items = sorted(obj.items(),
                       key=lambda kv: kv[0].encode("utf-16-be"))
        return "{" + ",".join(
            json.dumps(k, ensure_ascii=False) + ":" + _jcs(v)
            for k, v in items) + "}"
    raise TypeError(f"cannot canonicalize {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _stamp_url():
    return os.environ.get("ENTROPY_STAMP_URL", DEFAULT_STAMP_URL)


def available():
    """True if Stamp is reachable in either mode. Cheap check: local
    import only -- remote reachability is proven on first call."""
    srv, _ = _local_stamp()
    return srv is not None


def get_time(server="time.cloudflare.com"):
    """Current UTC time via Stamp.

    Returns {utc, ntp_server, offset_ms, stratum}. When NTP is
    unreachable the local clock is used and the ntp_* fields are null --
    matching Stamp's own degrade-and-disclose behavior.
    """
    srv, _ = _local_stamp()
    if srv is not None:
        try:
            ntp = srv.query_time(server)
            return {"utc": ntp["utc"], "ntp_server": ntp["server"],
                    "offset_ms": ntp["offset_ms"], "stratum": ntp["stratum"]}
        except Exception:
            pass  # fall through to local clock, disclosed via nulls
        return {"utc": datetime.now(timezone.utc).isoformat(),
                "ntp_server": None, "offset_ms": None, "stratum": None}
    try:
        data = _remote_call(_stamp_url(), "get_time", {"server": server})
        return {"utc": data["utc"], "ntp_server": data.get("server"),
                "offset_ms": data.get("offset_ms"),
                "stratum": data.get("stratum")}
    except StampUnavailable:
        raise
    except Exception as exc:
        raise StampUnavailable(f"get_time failed: {exc}") from exc


def generate_uuid(count=1):
    """One or more UUIDv4s via Stamp."""
    srv, _ = _local_stamp()
    if srv is not None:
        return [str(_uuid.uuid4()) for _ in range(count)]
    data = _remote_call(_stamp_url(), "generate_uuid", {"count": count})
    return data["uuids"]


def attest(payload):
    """Attest a payload via Stamp. Raises StampUnavailable if neither
    mode works -- callers must not degrade to unattested records."""
    srv, _ = _local_stamp()
    if srv is not None:
        return srv.attest(payload)
    return _remote_call(_stamp_url(), "attest", {"payload": payload})


def verify(record):
    """Verify a Stamp attestation record. Local verify is pure hash
    recomputation -- no network needed when stamp_mcp is installed."""
    srv, _ = _local_stamp()
    if srv is not None:
        return srv.verify(record)
    return _remote_call(_stamp_url(), "verify", {"attested": record})


def canonical_hash(payload):
    """sha256 of the RFC 8785 canonical form of payload.

    Local mode uses Stamp's canon module. Remote mode uses the local
    JCS fallback (Stamp's HTTP API exposes no canonical_hash tool);
    both implement the same public standard.
    """
    _, canon = _local_stamp()
    if canon is not None:
        return canon.canonical_hash(payload)
    return hashlib.sha256(_jcs(payload).encode("utf-8")).hexdigest()


def canonicalize(payload):
    """RFC 8785 canonical form of payload (local JCS fallback if needed)."""
    _, canon = _local_stamp()
    if canon is not None:
        return canon.canonicalize(payload)
    return _jcs(payload)

"""drand anchoring -- optional hardening for commitments.

Anchoring binds a commitment to a drand mainnet round: the round's
randomness was not yet published when the commitment was made, which
proves the commitment is not newer than it claims (and the round number
gives an externally-verifiable lower bound on the commit time).

This path must degrade gracefully. If the beacon is unreachable, the
commitment proceeds with beacon_anchor: null and a warning -- never a
failure. Hard 2-second timeout; a commitment never blocks on network
latency beyond that.

Uses urllib so the zero-dependency stdio core stays zero-dependency.
Base URL configurable via ENTROPY_BEACON_URL.
"""

import json
import os
import urllib.request

from entropy_mcp.errors import BeaconUnavailable
from entropy_mcp.schema import make_beacon_anchor

DEFAULT_BEACON_URL = "https://api.drand.sh"
BEACON_TIMEOUT = 2.0


def _base_url():
    return os.environ.get("ENTROPY_BEACON_URL", DEFAULT_BEACON_URL).rstrip("/")


def _get(path, timeout=BEACON_TIMEOUT):
    url = _base_url() + path
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise BeaconUnavailable(
            f"drand beacon unreachable at {url}: {exc}",
            expected="reachable drand HTTP API") from exc


def fetch_latest():
    """Fetch the current drand round. Returns a beacon_anchor block.

    Raises BeaconUnavailable on any failure -- callers decide whether
    that is fatal (it is not, for commit).
    """
    data = _get("/public/latest")
    try:
        return make_beacon_anchor(int(data["round"]), str(data["randomness"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise BeaconUnavailable(
            f"malformed drand response: {exc}",
            expected="{round: int, randomness: hex}") from exc


def fetch_round(round_number):
    """Fetch a specific drand round for verification."""
    data = _get(f"/public/{int(round_number)}")
    try:
        return make_beacon_anchor(int(data["round"]), str(data["randomness"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise BeaconUnavailable(
            f"malformed drand response: {exc}",
            expected="{round: int, randomness: hex}") from exc


def verify_anchor(anchor):
    """Re-fetch the anchored round and confirm the randomness matches.

    Returns True on match, False on mismatch, None when the beacon
    cannot be reached (verification is best-effort, never fatal).
    """
    if anchor is None:
        return None
    try:
        current = fetch_round(anchor["round"])
    except (BeaconUnavailable, KeyError, TypeError):
        return None
    return (current["round"] == anchor["round"]
            and current["randomness"] == anchor["randomness"])

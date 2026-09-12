"""Tier 0 -- CSPRNG primitives.

All randomness derives from os.urandom(). The `random` module is never
used. When a caller supplies a seed, draws come from a deterministic
hash-based DRBG instead (see DeterministicStream) so the result is
byte-identical across processes, machines, Python patch versions, and
OSes.

Deterministic stream specification (the compatibility contract):

    seed_bytes = 32 bytes. A 64-char hex string decodes directly; any
        other input is hashed to 32 bytes with SHA-256 first.
    block(i)   = SHA256(seed_bytes || i_be64)   for i = 0, 1, 2, ...
    stream     = block(0) || block(1) || block(2) || ...

    Bytes are consumed strictly in order. Per operation:

    randbelow(n):  let bits = (n-1).bit_length() and b = ceil(bits/8).
        Draw b bytes, interpret as a big-endian integer, mask off the
        high (8*b - bits) bits, reject and redraw if the value >= n.
        For n <= 1, consumes nothing and returns 0.
    uniform():     draw 8 bytes as a big-endian integer v, return
        (v + 0.5) / 2**64 -- a float in the OPEN interval (0, 1).
    random_int:    one randbelow(max - min + 1) per value, in order.
    shuffle:       Fisher-Yates; for i = n-1 down to 1, one
        randbelow(i + 1) per step.
    choose without replacement, unweighted: partial Fisher-Yates; for
        i = 0..k-1, one randbelow(n - i) per step.
    choose without replacement, weighted (Efraimidis-Spirakis): one
        uniform() per item, in item order; key_i = -ln(u_i) / w_i.
    choose with replacement: one uniform() per draw, bisected into the
        cumulative weight array (unweighted: one randbelow(n) per draw).
"""

import base64
import hashlib
import math
import os

from entropy_mcp.errors import ParameterError, SeedError

MAX_BYTES = 4096
MAX_COUNT = 10000
MAX_ITEMS = 10000


# ---------------------------------------------------------------------------
# Byte streams: OS CSPRNG and the deterministic seeded DRBG.
# ---------------------------------------------------------------------------

class OSStream:
    """Bytes straight from the OS CSPRNG. Non-deterministic by definition."""

    seeded = False

    def read(self, n):
        return os.urandom(n)


class DeterministicStream:
    """SHA256(seed || counter_be64) counter-mode DRBG.

    Fully specified, dependency-free, and byte-identical everywhere.
    NOT a general-purpose RNG for callers -- it exists so that a seeded
    draw can be re-executed by a verifier and produce the same bytes.
    """

    seeded = True

    def __init__(self, seed_bytes):
        if len(seed_bytes) != 32:
            raise SeedError(
                "deterministic stream requires a 32-byte seed",
                parameter="seed", expected="32 bytes (64 hex chars)")
        self._seed = seed_bytes
        self._counter = 0
        self._buf = bytearray()

    def read(self, n):
        while len(self._buf) < n:
            block = hashlib.sha256(
                self._seed + self._counter.to_bytes(8, "big")).digest()
            self._counter += 1
            self._buf.extend(block)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out


def normalize_seed(seed):
    """Normalize a caller-supplied seed to 32 bytes.

    A 64-char hex string decodes directly. Any other string or bytes
    input is hashed to 32 bytes with SHA-256. Returns (seed_bytes,
    canonical_hex).
    """
    if isinstance(seed, str):
        if len(seed) == 64:
            try:
                raw = bytes.fromhex(seed)
                return raw, raw.hex()
            except ValueError:
                pass  # 64 chars but not hex: fall through to hashing
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        return digest, digest.hex()
    if isinstance(seed, (bytes, bytearray)):
        raw = bytes(seed)
        if len(raw) == 32:
            return raw, raw.hex()
        digest = hashlib.sha256(raw).digest()
        return digest, digest.hex()
    raise SeedError(
        "seed must be a hex string or bytes",
        parameter="seed",
        expected="64-char hex string, or any string/bytes (hashed to 32 "
                 "bytes with SHA-256)")


def get_stream(seed=None):
    """Return (stream, seed_hex). seed_hex is None for an OS stream."""
    if seed is None:
        return OSStream(), None
    seed_bytes, seed_hex = normalize_seed(seed)
    return DeterministicStream(seed_bytes), seed_hex


# ---------------------------------------------------------------------------
# Stream consumers: the only ways bytes leave a stream.
# ---------------------------------------------------------------------------

def randbelow(stream, n):
    """Uniform integer in [0, n) by rejection sampling. Never modulo.

    Draws the smallest number of bits that covers n, rejects values
    >= n, and redraws. For n <= 1, consumes nothing and returns 0.
    """
    if n <= 1:
        return 0
    bits = (n - 1).bit_length()
    nbytes = (bits + 7) // 8
    excess = 8 * nbytes - bits
    mask = (1 << bits) - 1
    while True:
        v = int.from_bytes(stream.read(nbytes), "big") & mask
        if v < n:
            return v


def uniform(stream):
    """Uniform float in the OPEN interval (0, 1).

    v = int.from_bytes(read(8), 'big'); returns (v + 0.5) / 2**64.
    The open interval matters: log(u) and 1/u must never see 0 or 1.
    """
    v = int.from_bytes(stream.read(8), "big")
    return (v + 0.5) / 18446744073709551616.0


# ---------------------------------------------------------------------------
# Public primitives
# ---------------------------------------------------------------------------

def random_bytes(count, encoding="hex"):
    """Raw bytes from the OS CSPRNG. Non-deterministic by definition."""
    if not isinstance(count, int) or isinstance(count, bool) \
            or not 1 <= count <= MAX_BYTES:
        raise ParameterError(
            f"count must be an integer between 1 and {MAX_BYTES}",
            parameter="count", expected=f"1..{MAX_BYTES}")
    if encoding not in ("hex", "base64"):
        raise ParameterError(
            f"encoding must be 'hex' or 'base64', got {encoding!r}",
            parameter="encoding", expected="hex | base64")
    raw = os.urandom(count)
    if encoding == "hex":
        encoded = raw.hex()
    else:
        encoded = base64.b64encode(raw).decode("ascii")
    return {
        "bytes": encoded,
        "count": count,
        "encoding": encoding,
        "source": "os.urandom",
    }


def random_int(min, max, count=1, seed=None):
    """Uniform integers in [min, max], inclusive, by rejection sampling.

    With a seed, draws come from the deterministic stream and are
    byte-identical across runs. Without one, from os.urandom.
    """
    for name, val in (("min", min), ("max", max), ("count", count)):
        if not isinstance(val, int) or isinstance(val, bool):
            raise ParameterError(f"{name} must be an integer",
                                 parameter=name, expected="integer")
    if min > max:
        raise ParameterError(
            f"min ({min}) must be <= max ({max})",
            parameter="min", expected="min <= max")
    if not 1 <= count <= MAX_COUNT:
        raise ParameterError(
            f"count must be between 1 and {MAX_COUNT}",
            parameter="count", expected=f"1..{MAX_COUNT}")
    stream, seed_hex = get_stream(seed)
    span = max - min + 1
    values = [min + randbelow(stream, span) for _ in range(count)]
    return {
        "values": values,
        "min": min,
        "max": max,
        "count": count,
        "seeded": stream.seeded,
        "seed": seed_hex,
    }


def shuffle(items, seed=None, return_permutation=True):
    """Fisher-Yates shuffle. permutation[i] is the ORIGINAL index of the
    element now at position i.

    Example: items [a, b, c] -> shuffled [c, a, b] has permutation
    [2, 0, 1]: position 0 holds the element that was at index 2.
    This direction is a common source of silent bugs -- it is tested.
    """
    if not isinstance(items, (list, tuple)) or not 1 <= len(items) <= MAX_ITEMS:
        raise ParameterError(
            f"items must be a list of 1..{MAX_ITEMS} elements",
            parameter="items", expected=f"list of 1..{MAX_ITEMS}")
    items = list(items)
    stream, seed_hex = get_stream(seed)
    n = len(items)
    perm = list(range(n))
    out = list(items)
    for i in range(n - 1, 0, -1):
        j = randbelow(stream, i + 1)
        out[i], out[j] = out[j], out[i]
        perm[i], perm[j] = perm[j], perm[i]
    result = {"shuffled": out, "seeded": stream.seeded}
    if seed_hex is not None:
        result["seed"] = seed_hex
    if return_permutation:
        result["permutation"] = perm
    return result


def choose(items, k=1, weights=None, replacement=False, seed=None):
    """Choose k items, optionally weighted, with or without replacement.

    - No weights, no replacement: partial Fisher-Yates.
    - Weights, no replacement: Efraimidis-Spirakis exponential sort --
      key_i = -ln(u_i)/w_i, take the k smallest keys.
    - With replacement: cumulative-weight binary search (unweighted:
      one randbelow per draw).

    `probabilities` reports each selected item's effective selection
    probability: for weighted draws, its weight share w_i/W (nominal --
    for without-replacement draws the true inclusion probability is
    order-dependent); for unweighted without replacement, the marginal
    k/n; for unweighted with replacement, 1/n.
    """
    if not isinstance(items, (list, tuple)) or not 1 <= len(items) <= MAX_ITEMS:
        raise ParameterError(
            f"items must be a list of 1..{MAX_ITEMS} elements",
            parameter="items", expected=f"list of 1..{MAX_ITEMS}")
    items = list(items)
    n = len(items)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ParameterError("k must be a positive integer",
                             parameter="k", expected="integer >= 1")
    if not replacement and k > n:
        raise ParameterError(
            f"k ({k}) exceeds len(items) ({n}) without replacement",
            parameter="k", expected="k <= len(items) when "
                                    "replacement is false")
    if weights is not None:
        if not isinstance(weights, (list, tuple)) or len(weights) != n:
            raise ParameterError(
                "weights must be a list the same length as items",
                parameter="weights", expected=f"list of {n} numbers")
        weights = list(weights)
        for i, w in enumerate(weights):
            if not isinstance(w, (int, float)) or isinstance(w, bool) \
                    or not math.isfinite(w) or w < 0:
                raise ParameterError(
                    f"weights[{i}] must be a non-negative finite number",
                    parameter="weights", expected="non-negative floats")
        if all(w == 0 for w in weights):
            raise ParameterError(
                "all weights are zero -- nothing can ever be selected",
                parameter="weights", expected="at least one weight > 0")

    stream, seed_hex = get_stream(seed)

    if weights is None and not replacement:
        # Partial Fisher-Yates: k draws, each randbelow(n - i).
        idx = list(range(n))
        chosen = []
        for i in range(k):
            j = i + randbelow(stream, n - i)
            idx[i], idx[j] = idx[j], idx[i]
            chosen.append(idx[i])
        probs = [k / n] * k
    elif weights is not None and not replacement:
        # Efraimidis-Spirakis: exponential race keys, k smallest win.
        keys = []
        for i in range(n):
            if weights[i] == 0:
                keys.append(math.inf)
            else:
                keys.append(-math.log(uniform(stream)) / weights[i])
        chosen = sorted(range(n), key=lambda i: keys[i])[:k]
        total = sum(weights)
        probs = [weights[i] / total for i in chosen]
    elif weights is not None:
        # With replacement: prefix sums + bisect.
        import bisect
        prefix = []
        run = 0.0
        for w in weights:
            run += w
            prefix.append(run)
        total = run
        chosen = []
        for _ in range(k):
            r = uniform(stream) * total
            chosen.append(bisect.bisect_left(prefix, r))
        probs = [weights[i] / total for i in chosen]
    else:
        chosen = [randbelow(stream, n) for _ in range(k)]
        probs = [1.0 / n] * k

    result = {
        "items": [items[i] for i in chosen],
        "indices": chosen,
        "probabilities": probs,
        "k": k,
        "replacement": replacement,
        "weighted": weights is not None,
        "seeded": stream.seeded,
    }
    if seed_hex is not None:
        result["seed"] = seed_hex
    return result

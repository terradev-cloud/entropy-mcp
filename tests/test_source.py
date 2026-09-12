"""Unit tests for tier 0 primitives."""

import math

import pytest

from entropy_mcp import source
from entropy_mcp.errors import ParameterError, SeedError


# ---------------------------------------------------------------------------
# random_bytes
# ---------------------------------------------------------------------------

def test_random_bytes_hex():
    r = source.random_bytes(16)
    assert r["count"] == 16
    assert r["encoding"] == "hex"
    assert r["source"] == "os.urandom"
    assert len(r["bytes"]) == 32
    bytes.fromhex(r["bytes"])  # valid hex


def test_random_bytes_base64():
    import base64
    r = source.random_bytes(16, "base64")
    assert len(base64.b64decode(r["bytes"])) == 16


def test_random_bytes_bounds():
    with pytest.raises(ParameterError):
        source.random_bytes(0)
    with pytest.raises(ParameterError):
        source.random_bytes(4097)
    with pytest.raises(ParameterError):
        source.random_bytes(16, "utf8")


# ---------------------------------------------------------------------------
# random_int
# ---------------------------------------------------------------------------

def test_random_int_inclusive_bounds():
    r = source.random_int(5, 5, count=10)
    assert r["values"] == [5] * 10
    assert r["seeded"] is False


def test_random_int_range():
    r = source.random_int(0, 9, count=1000)
    assert all(0 <= v <= 9 for v in r["values"])
    assert len(set(r["values"])) > 5  # not degenerate


def test_random_int_seeded_reproducible():
    a = source.random_int(0, 100, count=50, seed="ab" * 32)
    b = source.random_int(0, 100, count=50, seed="ab" * 32)
    assert a["values"] == b["values"]
    assert a["seeded"] is True
    assert a["seed"] == "ab" * 32


def test_random_int_different_seeds_differ():
    a = source.random_int(0, 10**9, count=20, seed="aa" * 32)
    b = source.random_int(0, 10**9, count=20, seed="bb" * 32)
    assert a["values"] != b["values"]


def test_random_int_errors():
    with pytest.raises(ParameterError):
        source.random_int(10, 5)
    with pytest.raises(ParameterError):
        source.random_int(0, 10, count=0)
    with pytest.raises(ParameterError):
        source.random_int(0, 10, count=10001)
    with pytest.raises(ParameterError):
        source.random_int(0.5, 10)


# ---------------------------------------------------------------------------
# Rejection sampling internals
# ---------------------------------------------------------------------------

def test_randbelow_no_modulo_bias_structure():
    """randbelow must consume whole blocks and reject, never mod."""
    stream = source.DeterministicStream(b"\x00" * 32)
    # n=1 consumes nothing
    assert source.randbelow(stream, 1) == 0
    # values stay in range over many draws
    stream2 = source.DeterministicStream(b"\x01" * 32)
    vals = [source.randbelow(stream2, 7) for _ in range(500)]
    assert all(0 <= v < 7 for v in vals)


def test_uniform_open_interval():
    stream = source.DeterministicStream(b"\x02" * 32)
    for _ in range(1000):
        u = source.uniform(stream)
        assert 0.0 < u < 1.0


# ---------------------------------------------------------------------------
# Seed normalization
# ---------------------------------------------------------------------------

def test_normalize_seed_hex64():
    raw, hexed = source.normalize_seed("ab" * 32)
    assert raw == bytes.fromhex("ab" * 32)
    assert hexed == "ab" * 32


def test_normalize_seed_short_string_hashed():
    import hashlib
    raw, hexed = source.normalize_seed("my-seed")
    assert raw == hashlib.sha256(b"my-seed").digest()
    assert hexed == raw.hex()


def test_normalize_seed_bytes():
    raw, hexed = source.normalize_seed(b"\x11" * 32)
    assert raw == b"\x11" * 32
    raw2, _ = source.normalize_seed(b"short")
    import hashlib
    assert raw2 == hashlib.sha256(b"short").digest()


def test_normalize_seed_rejects_bad_types():
    with pytest.raises(SeedError):
        source.normalize_seed(12345)
    with pytest.raises(SeedError):
        source.normalize_seed(None)


# ---------------------------------------------------------------------------
# shuffle
# ---------------------------------------------------------------------------

def test_shuffle_is_permutation():
    items = list(range(100))
    r = source.shuffle(items, seed="cc" * 32)
    assert sorted(r["shuffled"]) == items
    assert sorted(r["permutation"]) == items


def test_shuffle_permutation_direction():
    """permutation[i] = original index of element now at position i."""
    items = ["a", "b", "c", "d", "e"]
    r = source.shuffle(items, seed="dd" * 32)
    for pos, orig_idx in enumerate(r["permutation"]):
        assert r["shuffled"][pos] == items[orig_idx]


def test_shuffle_seeded_reproducible():
    a = source.shuffle(list(range(50)), seed="ee" * 32)
    b = source.shuffle(list(range(50)), seed="ee" * 32)
    assert a["shuffled"] == b["shuffled"]
    assert a["permutation"] == b["permutation"]


def test_shuffle_errors():
    with pytest.raises(ParameterError):
        source.shuffle([])
    with pytest.raises(ParameterError):
        source.shuffle("notalist")


# ---------------------------------------------------------------------------
# choose
# ---------------------------------------------------------------------------

def test_choose_unweighted_no_replacement():
    r = source.choose(list(range(10)), k=5, seed="ff" * 32)
    assert len(r["items"]) == 5
    assert len(set(r["indices"])) == 5  # no repeats
    assert all(0 <= i < 10 for i in r["indices"])


def test_choose_k_exceeds_items():
    with pytest.raises(ParameterError):
        source.choose([1, 2, 3], k=4, replacement=False)
    # with replacement it's fine
    r = source.choose([1, 2, 3], k=10, replacement=True, seed="11" * 32)
    assert len(r["items"]) == 10


def test_choose_weighted_no_replacement():
    items = ["a", "b", "c", "d"]
    weights = [0.0, 1.0, 1.0, 1.0]
    r = source.choose(items, k=3, weights=weights, seed="22" * 32)
    assert "a" not in r["items"]  # zero weight never selected
    assert len(set(r["indices"])) == 3


def test_choose_all_zero_weights_errors():
    with pytest.raises(ParameterError):
        source.choose([1, 2], weights=[0.0, 0.0])


def test_choose_weight_mismatch_errors():
    with pytest.raises(ParameterError):
        source.choose([1, 2, 3], weights=[1.0, 2.0])
    with pytest.raises(ParameterError):
        source.choose([1, 2], weights=[1.0, -1.0])


def test_choose_weighted_with_replacement():
    items = ["x", "y"]
    r = source.choose(items, k=100, weights=[0.9, 0.1],
                      replacement=True, seed="33" * 32)
    assert r["items"].count("x") > r["items"].count("y")


def test_choose_efraimidis_spirakis_respects_weights():
    """Heavier items should be selected first more often."""
    items = list(range(5))
    weights = [10.0, 1.0, 1.0, 1.0, 1.0]
    first_picks = 0
    trials = 200
    for i in range(trials):
        r = source.choose(items, k=1, weights=weights,
                          seed=f"{i:064x}")
        if r["indices"][0] == 0:
            first_picks += 1
    # P(item 0) = 10/14 ~ 0.71; allow generous slack
    assert first_picks > trials * 0.5


def test_choose_seeded_reproducible():
    kw = dict(k=3, weights=[0.1, 0.2, 0.3, 0.4], seed="44" * 32)
    a = source.choose(["w", "x", "y", "z"], **kw)
    b = source.choose(["w", "x", "y", "z"], **kw)
    assert a["indices"] == b["indices"]

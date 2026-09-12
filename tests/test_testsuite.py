"""Tests for the entropy test battery itself."""

import math
import os

import pytest

from entropy_mcp import testsuite as ts
from entropy_mcp.errors import ParameterError


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------

def test_gammaincc_known_values():
    # Q(1, x) = e^-x
    assert ts.gammaincc_q(1.0, 2.0) == pytest.approx(math.exp(-2), rel=1e-10)
    # Q(0.5, x) = erfc(sqrt(x))
    assert ts.gammaincc_q(0.5, 1.0) == pytest.approx(math.erfc(1.0),
                                                   rel=1e-10)
    # chi2 with df=2: p = e^(-x/2)
    assert ts.chi2_pvalue(4.0, 2) == pytest.approx(math.exp(-2), rel=1e-10)


def test_ks_pvalue_bounds():
    assert ts.ks_pvalue(0.0, 100) == pytest.approx(1.0)
    assert ts.ks_pvalue(0.5, 100) < 0.001


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def test_infer_numeric():
    kind, vals = ts._infer([1, 2, 3, 4], "auto")
    assert kind == "numeric"


def test_infer_categorical():
    kind, vals = ts._infer(["a", "b", "a"], "auto")
    assert kind == "categorical"


def test_infer_hex_bytes():
    kind, vals = ts._infer(os.urandom(32).hex(), "auto")
    assert kind == "bytes"
    assert len(vals) == 32


def test_bad_input():
    with pytest.raises(ParameterError):
        ts.test_entropy([1])
    with pytest.raises(ParameterError):
        ts.test_entropy("zzzz-not-hex", input_type="bytes")
    with pytest.raises(ParameterError):
        ts.test_entropy([1, 2, 3], tests=["nonexistent"])
    with pytest.raises(ParameterError):
        ts.test_entropy([1, 2, 3], alpha=1.5)


# ---------------------------------------------------------------------------
# Good entropy passes, bad entropy fails
# ---------------------------------------------------------------------------

def test_good_bytes_pass():
    # Deterministic stream: uniform bytes, reproducible -- os.urandom
    # here would flake ~5% of the time at alpha=0.05 by chance alone.
    from entropy_mcp import source
    seq = source.DeterministicStream(b"\x42" * 32).read(4096).hex()
    r = ts.test_entropy(seq, tests=["frequency", "chi_squared", "runs",
                                    "entropy", "gap"])
    assert r["input_type"] == "bytes"
    assert r["tests"]["frequency"]["passed"] is True
    assert r["tests"]["entropy"]["passed"] is True


def test_biased_bytes_fail():
    # All zeros except a sprinkling -- monobit must scream.
    seq = (b"\x00" * 4000 + b"\xff" * 10).hex()
    r = ts.test_entropy(seq, tests=["frequency"])
    assert r["tests"]["frequency"]["passed"] is False


def test_constant_sequence_fails():
    r = ts.test_entropy([7] * 500, tests=["runs", "serial"])
    assert r["tests"]["runs"]["passed"] is False
    assert r["tests"]["serial"]["passed"] is False


def test_categorical_biased_fails_chi2():
    seq = ["a"] * 900 + ["b"] * 50 + ["c"] * 50
    r = ts.test_entropy(seq, tests=["chi_squared"])
    assert r["tests"]["chi_squared"]["passed"] is False


def test_alternating_sequence_fails_runs_and_serial():
    seq = [0, 1] * 500
    r = ts.test_entropy(seq, tests=["runs", "serial"])
    assert r["tests"]["runs"]["passed"] is False
    assert r["tests"]["serial"]["passed"] is False


def test_ks_with_distribution():
    from entropy_mcp.distributions import sample
    xs = sample("normal", {"mean": 0, "sd": 1}, n=2000,
                seed="ab" * 32)["samples"]
    r = ts.test_entropy(xs, tests=["ks"], distribution="normal",
                        dist_params={"mean": 0, "sd": 1})
    assert r["tests"]["ks"]["passed"] is True
    # Wrong distribution should fail
    r2 = ts.test_entropy(xs, tests=["ks"], distribution="normal",
                         dist_params={"mean": 5, "sd": 1})
    assert r2["tests"]["ks"]["passed"] is False


def test_ks_skipped_without_distribution():
    r = ts.test_entropy([1.0, 2.0, 3.0, 4.0, 5.0], tests=["ks"])
    assert r["tests"]["ks"]["passed"] is None


def test_verdict_and_counts():
    r = ts.test_entropy(os.urandom(2048).hex())
    assert "verdict" in r
    assert r["tests_run"] > 0
    assert r["tests_passed"] <= r["tests_run"]

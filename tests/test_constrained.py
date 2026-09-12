"""Tests for attribute-constrained sampling."""

import pytest

from entropy_mcp import constrained
from entropy_mcp.errors import ParameterError

SEED = "cafe" * 16

ATTRS = {
    "role": {"engineer": 0.4, "designer": 0.3, "pm": 0.3},
    "region": {"emea": 0.5, "amer": 0.35, "apac": 0.15},
}


def test_basic_shape():
    r = constrained.sample_constrained(ATTRS, 500, seed=SEED)
    assert len(r["records"]) == 500
    for rec in r["records"]:
        assert set(rec) == {"role", "region"}
        assert rec["role"] in ATTRS["role"]
        assert rec["region"] in ATTRS["region"]


def test_conformance_block():
    r = constrained.sample_constrained(ATTRS, 2000, seed=SEED)
    for attr, block in r["conformance"].items():
        assert set(block) == {"target", "realized", "counts",
                              "chi_squared", "p_value",
                              "max_abs_deviation", "verdict"}
        assert sum(block["counts"].values()) == 2000
        assert block["verdict"] in ("conforms", "deviates")
        # at n=2000 realized should be close to target
        for val, p in block["target"].items():
            assert abs(block["realized"][val] - p) < 0.08


def test_proportions_must_sum_to_one():
    bad = {"role": {"a": 0.5, "b": 0.4}}
    with pytest.raises(ParameterError):
        constrained.sample_constrained(bad, 10)


def test_negative_proportion_rejected():
    bad = {"role": {"a": 1.5, "b": -0.5}}
    with pytest.raises(ParameterError):
        constrained.sample_constrained(bad, 10)


def test_seeded_reproducible():
    a = constrained.sample_constrained(ATTRS, 100, seed=SEED)
    b = constrained.sample_constrained(ATTRS, 100, seed=SEED)
    assert a["records"] == b["records"]


def test_independence():
    """Attributes must not correlate. With independent sampling,
    P(role=engineer | region=emea) ~= P(role=engineer)."""
    r = constrained.sample_constrained(ATTRS, 5000, seed=SEED)
    recs = r["records"]
    emea = [x for x in recs if x["region"] == "emea"]
    p_eng_given_emea = sum(1 for x in emea
                           if x["role"] == "engineer") / len(emea)
    assert abs(p_eng_given_emea - 0.4) < 0.05


def test_n_bounds():
    with pytest.raises(ParameterError):
        constrained.sample_constrained(ATTRS, 0)
    with pytest.raises(ParameterError):
        constrained.sample_constrained(ATTRS, 10001)

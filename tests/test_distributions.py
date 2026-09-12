"""Distribution tests: parameter validation, determinism, and
statistical validation against theory (fixed seed, not flaky)."""

import math

import pytest

from entropy_mcp import distributions as dist
from entropy_mcp.errors import ParameterError
from entropy_mcp.testsuite import ks_pvalue

SEED = "deadbeef" * 8


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_unknown_distribution():
    with pytest.raises(ParameterError):
        dist.sample("cauchy", {}, 1)


def test_param_validation():
    with pytest.raises(ParameterError):
        dist.sample("normal", {"mean": 0, "sd": 0})
    with pytest.raises(ParameterError):
        dist.sample("normal", {"mean": 0, "sd": -1})
    with pytest.raises(ParameterError):
        dist.sample("exponential", {"rate": 0})
    with pytest.raises(ParameterError):
        dist.sample("uniform", {"low": 5, "high": 5})
    with pytest.raises(ParameterError):
        dist.sample("uniform", {"low": 5, "high": 2})
    with pytest.raises(ParameterError):
        dist.sample("triangular", {"low": 0, "mode": 5, "high": 2})
    with pytest.raises(ParameterError):
        dist.sample("triangular", {"low": 0, "mode": 3, "high": 2})
    with pytest.raises(ParameterError):
        dist.sample("beta", {"alpha": 0, "beta": 1})
    with pytest.raises(ParameterError):
        dist.sample("beta", {"alpha": 1, "beta": -1})
    with pytest.raises(ParameterError):
        dist.sample("uniform", {"low": 0}, 1)  # missing high


def test_n_bounds():
    with pytest.raises(ParameterError):
        dist.sample("uniform", {"low": 0, "high": 1}, n=0)
    with pytest.raises(ParameterError):
        dist.sample("uniform", {"low": 0, "high": 1}, n=10001)


def test_seeded_reproducible():
    a = dist.sample("normal", {"mean": 0, "sd": 1}, n=20, seed=SEED)
    b = dist.sample("normal", {"mean": 0, "sd": 1}, n=20, seed=SEED)
    assert a["samples"] == b["samples"]


def test_box_muller_uses_both_variates():
    """Odd n still consumes the full final pair: n and n+1 share the
    first n samples exactly."""
    a = dist.sample("normal", {"mean": 0, "sd": 1}, n=5, seed=SEED)
    b = dist.sample("normal", {"mean": 0, "sd": 1}, n=6, seed=SEED)
    assert a["samples"] == b["samples"][:5]


def test_summary_block():
    r = dist.sample("uniform", {"low": 0, "high": 1}, n=100, seed=SEED)
    s = r["summary"]
    for key in ("mean", "sd", "min", "max", "p5", "p50", "p95"):
        assert key in s
    assert s["min"] <= s["p5"] <= s["p50"] <= s["p95"] <= s["max"]


# ---------------------------------------------------------------------------
# Statistical validation: 100k samples, mean/variance within 3 standard
# errors, KS vs theoretical CDF at alpha=0.01. Fixed seed => not flaky.
# ---------------------------------------------------------------------------

N = 100_000


def _draw(name, params, n, seed):
    """Draw n samples in batches of MAX_N with derived seeds -- the
    per-call cap is 10k, and determinism is preserved because each
    batch seed is seed + batch index."""
    xs = []
    for i in range(0, n, dist.MAX_N):
        batch = min(dist.MAX_N, n - i)
        xs.extend(dist.sample(name, params, n=batch,
                              seed=f"{seed}:{i}")["samples"])
    return xs


def _check(name, params, mean, var, n=N):
    xs = _draw(name, params, n, SEED)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    se_mean = math.sqrt(var / n)
    # SE of variance estimate ~ sqrt(2 var^2 / n) for near-normal; use a
    # generous bound for skewed distributions.
    se_var = math.sqrt(2 * var * var / n) * 3
    assert abs(m - mean) < 3 * se_mean, \
        f"{name}: mean {m} vs theory {mean} (se {se_mean})"
    assert abs(v - var) < 3 * se_var, \
        f"{name}: var {v} vs theory {var} (se {se_var})"
    # KS against theoretical CDF
    ordered = sorted(xs)
    d = 0.0
    for i, x in enumerate(ordered):
        f = dist.cdf(name, params, x)
        d = max(d, abs((i + 1) / n - f), abs(f - i / n))
    p = ks_pvalue(d, n)
    assert p > 0.01, f"{name}: KS p={p}"


def test_uniform_stats():
    _check("uniform", {"low": 2, "high": 5}, mean=3.5, var=9 / 12)


def test_normal_stats():
    _check("normal", {"mean": 10, "sd": 2}, mean=10.0, var=4.0)


def test_lognormal_stats():
    mu, sigma = 0.5, 0.4
    mean = math.exp(mu + sigma ** 2 / 2)
    var = (math.exp(sigma ** 2) - 1) * math.exp(2 * mu + sigma ** 2)
    _check("lognormal", {"mu": mu, "sigma": sigma}, mean=mean, var=var)


def test_exponential_stats():
    _check("exponential", {"rate": 2.0}, mean=0.5, var=0.25)


def test_triangular_stats():
    low, mode, high = 0, 1, 4
    mean = (low + mode + high) / 3
    var = (low ** 2 + mode ** 2 + high ** 2
           - low * mode - low * high - mode * high) / 18
    _check("triangular", {"low": low, "mode": mode, "high": high},
           mean=mean, var=var)


def test_beta_stats_bb():
    a, b = 2.0, 5.0
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    _check("beta", {"alpha": a, "beta": b}, mean=mean, var=var)


def test_beta_stats_bc():
    """BC path: min <= 1 < max."""
    a, b = 0.5, 3.0
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    _check("beta", {"alpha": a, "beta": b}, mean=mean, var=var)


def test_beta_stats_johnk():
    """Johnk path: both < 1."""
    a, b = 0.5, 0.7
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    _check("beta", {"alpha": a, "beta": b}, mean=mean, var=var)


# ---------------------------------------------------------------------------
# CDF sanity
# ---------------------------------------------------------------------------

def test_cdf_endpoints():
    assert dist.cdf("uniform", {"low": 0, "high": 1}, -1) == 0.0
    assert dist.cdf("uniform", {"low": 0, "high": 1}, 2) == 1.0
    assert dist.cdf("normal", {"mean": 0, "sd": 1}, 0) == pytest.approx(0.5)
    assert dist.cdf("exponential", {"rate": 1}, 0) == 0.0
    assert dist.cdf("beta", {"alpha": 2, "beta": 2}, 0.5) == \
        pytest.approx(0.5)
    assert dist.cdf("triangular", {"low": 0, "mode": 1, "high": 2},
                    1) == pytest.approx(0.5)

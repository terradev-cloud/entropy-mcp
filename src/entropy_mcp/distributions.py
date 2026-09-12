"""Tier 2 -- named distributions.

Six distributions, pure stdlib `math`, no numpy/scipy. Every method
consumes uniforms from the byte stream in a fixed, documented order so
a seeded draw reproduces byte-identically.

Stream consumption order per distribution (per sample unless noted):

    uniform:      1 uniform -> low + (high - low) * u
    normal:       Box-Muller in PAIRS -- draw u1 then u2, emit
                  z0 = sqrt(-2 ln u1) cos(2 pi u2) first, then
                  z1 = sqrt(-2 ln u1) sin(2 pi u2). For odd n the final
                  pair is still fully consumed; z1 is computed and
                  discarded. Both variates are always used in order.
    lognormal:    exp(normal(mu, sigma)) -- same consumption as normal.
    exponential:  1 uniform -> -ln(u) / rate
    triangular:   1 uniform -> inverse CDF
    beta:         rejection samplers, 2 uniforms per trial:
                  - Johnk (alpha < 1 and beta < 1): x = u1^(1/a),
                    y = u2^(1/b); accept when x + y <= 1, deliver
                    x / (x + y).
                  - Cheng BB (min(a,b) > 1) and Cheng BC (min <= 1,
                    max >= 1): per R. C. H. Cheng, CACM 21(4), 1978,
                    as implemented in R's rbeta(). Two uniforms per
                    trial, u1 then u2.
"""

import math

from entropy_mcp.errors import ParameterError
from entropy_mcp.source import get_stream, uniform

MAX_N = 10000

DISTRIBUTIONS = ("uniform", "normal", "lognormal", "exponential",
                 "triangular", "beta")


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def _num(params, name, lo=None, hi=None, lo_open=False):
    """Pull a finite number out of params, range-checked."""
    v = params.get(name)
    if not isinstance(v, (int, float)) or isinstance(v, bool) \
            or not math.isfinite(v):
        raise ParameterError(
            f"{name} must be a finite number, got {v!r}",
            parameter=name, expected="finite number")
    v = float(v)
    if lo is not None and (v <= lo if lo_open else v < lo):
        op = ">" if lo_open else ">="
        raise ParameterError(
            f"{name} must be {op} {lo}, got {v}",
            parameter=name, expected=f"{op} {lo}")
    if hi is not None and v > hi:
        raise ParameterError(
            f"{name} must be <= {hi}, got {v}",
            parameter=name, expected=f"<= {hi}")
    return v


def _validate(distribution, params):
    """Validate params for a distribution; return a clean params dict."""
    if not isinstance(params, dict):
        raise ParameterError("params must be an object",
                             parameter="params", expected="object")
    if distribution == "uniform":
        low = _num(params, "low")
        high = _num(params, "high")
        if not low < high:
            raise ParameterError(
                f"low ({low}) must be < high ({high})",
                parameter="low", expected="low < high")
        return {"low": low, "high": high}
    if distribution == "normal":
        return {"mean": _num(params, "mean"),
                "sd": _num(params, "sd", lo=0, lo_open=True)}
    if distribution == "lognormal":
        return {"mu": _num(params, "mu"),
                "sigma": _num(params, "sigma", lo=0, lo_open=True)}
    if distribution == "exponential":
        return {"rate": _num(params, "rate", lo=0, lo_open=True)}
    if distribution == "triangular":
        low = _num(params, "low")
        mode = _num(params, "mode")
        high = _num(params, "high")
        if not low < high:
            raise ParameterError(
                f"low ({low}) must be < high ({high})",
                parameter="low", expected="low < high")
        if not low <= mode <= high:
            raise ParameterError(
                f"mode ({mode}) must satisfy low <= mode <= high",
                parameter="mode", expected="low <= mode <= high")
        return {"low": low, "mode": mode, "high": high}
    if distribution == "beta":
        return {"alpha": _num(params, "alpha", lo=0, lo_open=True),
                "beta": _num(params, "beta", lo=0, lo_open=True)}
    raise ParameterError(
        f"unknown distribution {distribution!r}",
        parameter="distribution",
        expected=" | ".join(DISTRIBUTIONS))


# ---------------------------------------------------------------------------
# Samplers. Each takes a stream and returns one variate.
# ---------------------------------------------------------------------------

def _s_uniform(stream, p):
    return p["low"] + (p["high"] - p["low"]) * uniform(stream)


def _normal_pair(stream):
    """Box-Muller: consume u1 then u2, return (z0, z1). Both are used."""
    u1 = uniform(stream)
    u2 = uniform(stream)
    r = math.sqrt(-2.0 * math.log(u1))
    return r * math.cos(2.0 * math.pi * u2), r * math.sin(2.0 * math.pi * u2)


def _s_exponential(stream, p):
    return -math.log(uniform(stream)) / p["rate"]


def _s_triangular(stream, p):
    low, mode, high = p["low"], p["mode"], p["high"]
    u = uniform(stream)
    c = (mode - low) / (high - low)
    if u < c:
        return low + math.sqrt(u * (high - low) * (mode - low))
    return high - math.sqrt((1.0 - u) * (high - low) * (high - mode))


def _beta_johnk(stream, a, b):
    """Johnk's algorithm for alpha < 1 and beta < 1. Two uniforms/trial."""
    inv_a = 1.0 / a
    inv_b = 1.0 / b
    while True:
        x = uniform(stream) ** inv_a
        y = uniform(stream) ** inv_b
        s = x + y
        if 0.0 < s <= 1.0:
            return x / s


def _beta_cheng(stream, alpha_p, beta_p):
    """Cheng's BB (min > 1) or BC (min <= 1) algorithm.

    Per R. C. H. Cheng, "Generating beta variates with nonintegral
    shape parameters", CACM 21(4):317-322, 1978 -- same construction as
    R's rbeta(). Two uniforms per trial, u1 then u2.
    """
    a = min(alpha_p, beta_p)
    b = max(alpha_p, beta_p)
    alpha = a + b
    a_is_alpha = alpha_p == a

    if a > 1.0:
        # --- Algorithm BB ---
        beta_alg = math.sqrt((alpha - 2.0) / (2.0 * a * b - alpha))
        gamma = a + 1.0 / beta_alg
        while True:
            u1 = uniform(stream)
            u2 = uniform(stream)
            v = beta_alg * math.log(u1 / (1.0 - u1))
            w = a * math.exp(v)
            if not math.isfinite(w):
                w = 1.7976931348623157e308
            z = u1 * u1 * u2
            r = gamma * v - 1.3862944  # log(4)
            s = a + r - w
            if s + 2.609438 >= 5.0 * z:  # 2.609438 = 1 + log(5)
                break
            t = math.log(z)
            if s > t:
                break
            if r + alpha * math.log(alpha / (b + w)) >= t:
                break
        return w / (b + w) if a_is_alpha else b / (b + w)

    # --- Algorithm BC ---
    beta_alg = 1.0 / a
    delta = 1.0 + b - a
    k1 = delta * (0.0138889 + 0.0416667 * a) / (b * beta_alg - 0.777778)
    k2 = 0.25 + (0.5 + 0.25 / delta) * a
    while True:
        u1 = uniform(stream)
        u2 = uniform(stream)
        if u1 < 0.5:
            y = u1 * u2
            z = u1 * y
            if 0.25 * u2 + z - y >= k1:
                continue
        else:
            z = u1 * u1 * u2
            if z <= 0.25:
                v = beta_alg * math.log(u1 / (1.0 - u1))
                w = b * math.exp(v)
                return a / (a + w) if a_is_alpha else w / (a + w)
            if z >= k2:
                continue
        v = beta_alg * math.log(u1 / (1.0 - u1))
        w = b * math.exp(v)
        if alpha * (math.log(alpha / (a + w)) + v) - 1.3862944 \
                >= math.log(z):
            return a / (a + w) if a_is_alpha else w / (a + w)


def _s_beta(stream, p):
    a, b = p["alpha"], p["beta"]
    if a < 1.0 and b < 1.0:
        return _beta_johnk(stream, a, b)
    return _beta_cheng(stream, a, b)


# ---------------------------------------------------------------------------
# Summary statistics: Welford single pass for mean/sd, sort for quantiles.
# ---------------------------------------------------------------------------

def _summary(samples):
    n = len(samples)
    mean = 0.0
    m2 = 0.0
    for i, x in enumerate(samples, 1):
        delta = x - mean
        mean += delta / i
        m2 += delta * (x - mean)
    sd = math.sqrt(m2 / (n - 1)) if n > 1 else 0.0
    ordered = sorted(samples)

    def q(p):
        if n == 1:
            return ordered[0]
        pos = p * (n - 1)
        lo = int(pos)
        frac = pos - lo
        if lo + 1 >= n:
            return ordered[-1]
        return ordered[lo] + frac * (ordered[lo + 1] - ordered[lo])

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "min": ordered[0],
        "max": ordered[-1],
        "p5": q(0.05),
        "p50": q(0.50),
        "p95": q(0.95),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sample(distribution, params, n=1, seed=None):
    """Draw n samples from a named distribution.

    Returns samples, the validated params, the seed used, and a summary
    block (mean, sd, min, max, p5, p50, p95). params_canonical_hash is
    added by the caller (commit.py/server.py) via Stamp -- this module
    stays dependency-free.
    """
    if distribution not in DISTRIBUTIONS:
        raise ParameterError(
            f"unknown distribution {distribution!r}",
            parameter="distribution",
            expected=" | ".join(DISTRIBUTIONS))
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_N:
        raise ParameterError(
            f"n must be an integer between 1 and {MAX_N}",
            parameter="n", expected=f"1..{MAX_N}")
    clean = _validate(distribution, params)
    stream, seed_hex = get_stream(seed)

    if distribution in ("normal", "lognormal"):
        # Box-Muller emits pairs; for odd n the last z1 is consumed and
        # discarded -- the stream position stays deterministic.
        samples = []
        pairs = (n + 1) // 2
        for _ in range(pairs):
            z0, z1 = _normal_pair(stream)
            samples.append(z0)
            samples.append(z1)
        samples = samples[:n]
        if distribution == "normal":
            samples = [clean["mean"] + clean["sd"] * z for z in samples]
        else:
            samples = [math.exp(clean["mu"] + clean["sigma"] * z)
                       for z in samples]
    else:
        sampler = {
            "uniform": _s_uniform,
            "exponential": _s_exponential,
            "triangular": _s_triangular,
            "beta": _s_beta,
        }[distribution]
        samples = [sampler(stream, clean) for _ in range(n)]

    result = {
        "distribution": distribution,
        "params": clean,
        "n": n,
        "samples": samples,
        "summary": _summary(samples),
        "seeded": stream.seeded,
    }
    if seed_hex is not None:
        result["seed"] = seed_hex
    return result


# ---------------------------------------------------------------------------
# Theoretical CDFs -- used by testsuite.py's Kolmogorov-Smirnov test and
# by the statistical validation tests. Pure math, no dependencies.
# ---------------------------------------------------------------------------

def _norm_cdf(x, mean=0.0, sd=1.0):
    return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function
    (Numerical Recipes betacf)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _beta_cdf(x, a, b):
    """Regularized incomplete beta I_x(a,b) -- Numerical Recipes betai.

    Uses the symmetry transform: below the threshold evaluate
    betacf(a,b,x), above it evaluate betacf(b,a,1-x).
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def cdf(distribution, params, x):
    """Theoretical CDF F(x) for a supported distribution."""
    p = _validate(distribution, params)
    if distribution == "uniform":
        if x <= p["low"]:
            return 0.0
        if x >= p["high"]:
            return 1.0
        return (x - p["low"]) / (p["high"] - p["low"])
    if distribution == "normal":
        return _norm_cdf(x, p["mean"], p["sd"])
    if distribution == "lognormal":
        if x <= 0:
            return 0.0
        return _norm_cdf(math.log(x), p["mu"], p["sigma"])
    if distribution == "exponential":
        if x <= 0:
            return 0.0
        return 1.0 - math.exp(-p["rate"] * x)
    if distribution == "triangular":
        low, mode, high = p["low"], p["mode"], p["high"]
        if x <= low:
            return 0.0
        if x >= high:
            return 1.0
        if x <= mode:
            return (x - low) ** 2 / ((high - low) * (mode - low))
        return 1.0 - (high - x) ** 2 / ((high - low) * (high - mode))
    if distribution == "beta":
        return _beta_cdf(x, p["alpha"], p["beta"])
    raise ParameterError(f"unknown distribution {distribution!r}",
                         parameter="distribution",
                         expected=" | ".join(DISTRIBUTIONS))

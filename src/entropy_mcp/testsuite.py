"""Entropy statistical tests -- informed by NIST SP 800-22, in the
spirit of it. No NIST certification is claimed or implied.

Accepts a list of numbers, a list of strings (categorical), or a
hex/base64 byte string. Type is inferred unless input_type is given.

Tests (all pure stdlib math):

    frequency       bit balance (bytes), above/below median (numeric),
                    chi-squared vs uniform (categorical)
    chi_squared     goodness of fit vs uniform or a supplied `expected`
    runs            Wald-Wolfowitz runs above/below median
    longest_run     longest run of identical values vs expectation
    serial          lag-1 autocorrelation (numeric/bytes) or transition
                    chi-squared (categorical)
    entropy         Shannon bits/symbol vs theoretical max
    ks              one-sample Kolmogorov-Smirnov vs a supplied
                    continuous distribution (numeric only; skipped
                    unless `distribution` + `dist_params` are given)
    gap             distribution of gaps between recurrences of the
                    most common symbol vs geometric expectation

Special functions implemented here: regularized incomplete gamma
(series / continued fraction split, Numerical Recipes), the Kolmogorov
asymptotic series, and the normal CDF via math.erf.
"""

import base64
import math
from collections import Counter

from entropy_mcp.errors import ParameterError

ALL_TESTS = ("frequency", "chi_squared", "runs", "longest_run",
             "serial", "entropy", "ks", "gap")


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------

def gammaincc_q(a, x):
    """Regularized upper incomplete gamma Q(a, x) = Gamma(a,x)/Gamma(a).

    Series expansion for x < a+1, continued fraction otherwise
    (Numerical Recipes gser/gcf). Used for chi-squared p-values.
    """
    if x < 0 or a <= 0:
        raise ValueError("gammaincc_q requires x >= 0, a > 0")
    if x == 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        # Series for P(a,x); return 1 - P.
        ap = a
        s = 1.0 / a
        delta = s
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            s += delta
            if abs(delta) < abs(s) * 1e-15:
                break
        p = s * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - p))
    # Continued fraction for Q(a,x).
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = h * math.exp(-x + a * math.log(x) - gln)
    return max(0.0, min(1.0, q))


def chi2_pvalue(stat, df):
    """p-value for a chi-squared statistic with df degrees of freedom."""
    if df <= 0:
        return None
    return gammaincc_q(df / 2.0, stat / 2.0)


def norm_sf(z):
    """Two-sided normal p-value for a z statistic: P(|Z| > |z|)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def ks_pvalue(d, n):
    """Kolmogorov asymptotic p-value for a one-sample KS statistic.

    Uses the Stephens effective-sample-size correction, then
    Q(lambda) = 2 * sum_k (-1)^(k-1) exp(-2 k^2 lambda^2).
    """
    if n <= 0:
        return None
    lam = d * (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n))
    s = 0.0
    for k in range(1, 200):
        term = (-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
        s += term
        if abs(term) < 1e-12:
            break
    return max(0.0, min(1.0, 2.0 * s))


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def _infer(sequence, input_type):
    """Normalize input to (kind, values) where kind is
    'numeric' | 'categorical' | 'bytes'."""
    if input_type not in (None, "auto", "numeric", "categorical", "bytes"):
        raise ParameterError(
            f"input_type must be auto|numeric|categorical|bytes, "
            f"got {input_type!r}", parameter="input_type",
            expected="auto | numeric | categorical | bytes")
    if isinstance(sequence, str):
        if input_type in (None, "auto", "bytes"):
            try:
                return "bytes", list(bytes.fromhex(sequence))
            except ValueError:
                pass
            try:
                return "bytes", list(base64.b64decode(sequence,
                                                      validate=True))
            except Exception:
                pass
            raise ParameterError(
                "string input is neither valid hex nor base64",
                parameter="sequence", expected="hex or base64 string")
        raise ParameterError("string input requires input_type 'bytes'",
                             parameter="input_type", expected="bytes")
    if not isinstance(sequence, (list, tuple)) or len(sequence) < 2:
        raise ParameterError(
            "sequence must be a list of at least 2 elements",
            parameter="sequence", expected="list, len >= 2")
    if all(isinstance(x, bool) for x in sequence):
        return "categorical", [str(x) for x in sequence]
    if all(isinstance(x, (int, float)) and math.isfinite(x)
           for x in sequence):
        if input_type == "categorical":
            return "categorical", [str(x) for x in sequence]
        return "numeric", [float(x) for x in sequence]
    return "categorical", [str(x) for x in sequence]


def _bits(values):
    out = []
    for b in values:
        for i in range(7, -1, -1):
            out.append((b >> i) & 1)
    return out


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _binary_view(kind, values):
    """Reduce any input kind to a binary list for runs-type tests.

    bytes -> bits; numeric -> above/below median; categorical -> is the
    most common category or not.
    """
    if kind == "bytes":
        return _bits(values), 0.5
    if kind == "numeric":
        med = _median(values)
        bits = [1 if x > med else 0 for x in values]
        p = sum(bits) / len(bits) if bits else 0.5
        return bits, p
    top = Counter(values).most_common(1)[0][0]
    bits = [1 if v == top else 0 for v in values]
    return bits, sum(bits) / len(bits)


# ---------------------------------------------------------------------------
# Individual tests. Each returns {statistic, p_value, passed, reading}.
# ---------------------------------------------------------------------------

def _t_frequency(kind, values, alpha, expected=None):
    if kind == "bytes":
        bits = _bits(values)
        n = len(bits)
        s = abs(2 * sum(bits) - n) / math.sqrt(n)
        p = math.erfc(s / math.sqrt(2.0))
        return {"statistic": s, "p_value": p, "passed": p > alpha,
                "reading": f"bit balance {'is' if p > alpha else 'is not'} "
                           f"consistent with a fair bit source "
                           f"({sum(bits)}/{n} ones)"}
    if kind == "categorical":
        counts = Counter(values)
        k = len(counts)
        n = len(values)
        exp = n / k
        stat = sum((c - exp) ** 2 / exp for c in counts.values())
        p = chi2_pvalue(stat, k - 1)
        return {"statistic": stat, "p_value": p, "passed": p > alpha,
                "reading": f"category frequencies {'are' if p > alpha else 'are not'} "
                           f"consistent with uniform over {k} symbols"}
    # numeric: above/below median balance
    bits, _ = _binary_view(kind, values)
    n = len(bits)
    s = abs(2 * sum(bits) - n) / math.sqrt(n)
    p = math.erfc(s / math.sqrt(2.0))
    return {"statistic": s, "p_value": p, "passed": p > alpha,
            "reading": f"above/below-median balance {'is' if p > alpha else 'is not'} "
                       f"consistent with a symmetric source"}


def _t_chi_squared(kind, values, alpha, expected=None):
    counts = Counter(values)
    n = len(values)
    if expected is not None:
        if not isinstance(expected, dict):
            raise ParameterError("expected must be an object mapping "
                                 "symbol -> probability",
                                 parameter="expected",
                                 expected="{symbol: probability}")
        total = sum(expected.values())
        if total <= 0:
            raise ParameterError("expected probabilities must sum > 0",
                                 parameter="expected")
        probs = {str(k): v / total for k, v in expected.items()}
        stat = 0.0
        syms = set(probs) | set(counts)
        for sym in syms:
            e = probs.get(sym, 0.0) * n
            o = counts.get(sym, 0)
            if e > 0:
                stat += (o - e) ** 2 / e
            elif o > 0:
                stat += o  # observed mass where none expected
        df = len(syms) - 1
    else:
        k = len(counts)
        if k < 2:
            return {"statistic": 0.0, "p_value": 0.0, "passed": False,
                    "reading": "single distinct symbol -- zero entropy"}
        e = n / k
        stat = sum((c - e) ** 2 / e for c in counts.values())
        df = k - 1
    p = chi2_pvalue(stat, df)
    return {"statistic": stat, "p_value": p,
            "passed": p is not None and p > alpha,
            "reading": f"symbol frequencies {'fit' if (p is not None and p > alpha) else 'do not fit'} "
                       f"the expected distribution (df={df})"}


def _t_runs(kind, values, alpha, expected=None):
    bits, _ = _binary_view(kind, values)
    n = len(bits)
    n1 = sum(bits)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return {"statistic": None, "p_value": 0.0, "passed": False,
                "reading": "sequence is constant -- zero runs variance"}
    runs = 1 + sum(1 for i in range(1, n) if bits[i] != bits[i - 1])
    mu = 2.0 * n1 * n0 / n + 1.0
    var = (2.0 * n1 * n0 * (2.0 * n1 * n0 - n)) / (n * n * (n - 1.0))
    if var <= 0:
        return {"statistic": None, "p_value": None, "passed": None,
                "reading": "too little variation for a runs test"}
    z = (runs - mu) / math.sqrt(var)
    p = norm_sf(z)
    return {"statistic": z, "p_value": p, "passed": p > alpha,
            "reading": f"{runs} runs {'is' if p > alpha else 'is not'} "
                       f"consistent with independence (expected ~{mu:.1f})"}


def _t_longest_run(kind, values, alpha, expected=None):
    bits, p_sym = _binary_view(kind, values)
    n = len(bits)
    longest = 1
    cur = 1
    for i in range(1, n):
        if bits[i] == bits[i - 1]:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    # Approximate expectation for the longest run of the dominant symbol:
    # E[L] ~ log_{1/p}(n(1-p)) + Euler-Mascheroni/ln(1/p) - 1/2.
    p = max(min(p_sym, 1.0 - p_sym), 1e-9)
    p_dom = 1.0 - p
    if p_dom <= 0 or p_dom >= 1:
        return {"statistic": float(longest), "p_value": None,
                "passed": None,
                "reading": "degenerate symbol balance -- test not applicable"}
    e_l = (math.log(n * (1.0 - p_dom)) / math.log(1.0 / p_dom)
           + 0.5772156649 / math.log(1.0 / p_dom) - 0.5)
    # Approximate two-sided check via the geometric tail: probability of
    # a run this long or longer somewhere in n positions.
    p_long = min(1.0, n * (1.0 - p_dom) * p_dom ** longest)
    p_short = min(1.0, math.exp(-n * (1.0 - p_dom)
                                * p_dom ** max(longest - 1, 0)))
    pval = min(1.0, 2.0 * min(p_long, p_short))
    return {"statistic": float(longest), "p_value": pval,
            "passed": pval > alpha,
            "reading": f"longest run {longest} vs expected ~{e_l:.1f} "
                       f"{'looks' if pval > alpha else 'does not look'} "
                       f"typical (approximate test)"}


def _t_serial(kind, values, alpha, expected=None):
    if kind == "categorical":
        # Chi-squared independence test on consecutive pairs.
        pairs = Counter(zip(values, values[1:]))
        rows = Counter(values[:-1])
        cols = Counter(values[1:])
        n = len(values) - 1
        stat = 0.0
        for (a, b), o in pairs.items():
            e = rows[a] * cols[b] / n
            if e > 0:
                stat += (o - e) ** 2 / e
        df = (len(rows) - 1) * (len(cols) - 1)
        p = chi2_pvalue(stat, df) if df > 0 else None
        return {"statistic": stat, "p_value": p,
                "passed": (p is None) or p > alpha,
                "reading": f"consecutive-symbol transitions {'look' if (p is None or p > alpha) else 'do not look'} "
                           f"independent"}
    xs = [float(v) for v in values]
    n = len(xs)
    mean = sum(xs) / n
    denom = sum((x - mean) ** 2 for x in xs)
    if denom == 0:
        return {"statistic": None, "p_value": 0.0, "passed": False,
                "reading": "sequence is constant -- autocorrelation "
                           "undefined"}
    num = sum((xs[i] - mean) * (xs[i + 1] - mean) for i in range(n - 1))
    r = num / denom
    z = r * math.sqrt(n)
    p = norm_sf(z)
    return {"statistic": r, "p_value": p, "passed": p > alpha,
            "reading": f"lag-1 autocorrelation {r:+.4f} {'is' if p > alpha else 'is not'} "
                       f"consistent with zero"}


def _t_entropy(kind, values, alpha, expected=None):
    counts = Counter(values)
    n = len(values)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    h_max = math.log2(len(counts)) if len(counts) > 1 else 0.0
    # p-value: chi-squared uniformity of the symbol distribution -- the
    # operational meaning of "is the entropy near its maximum".
    k = len(counts)
    e = n / k
    stat = sum((c - e) ** 2 / e for c in counts.values())
    p = chi2_pvalue(stat, k - 1) if k > 1 else None
    ratio = h / h_max if h_max > 0 else 0.0
    return {"statistic": h, "p_value": p,
            "passed": (p is None) or p > alpha,
            "reading": f"{h:.3f} bits/symbol vs max {h_max:.3f} "
                       f"({ratio:.1%} of theoretical max)",
            "bits_per_symbol": h, "max_bits": h_max}


def _t_ks(kind, values, alpha, expected=None, distribution=None,
         dist_params=None):
    if kind != "numeric":
        return {"statistic": None, "p_value": None, "passed": None,
                "reading": "KS test applies to numeric input only -- "
                           "skipped"}
    if distribution is None:
        return {"statistic": None, "p_value": None, "passed": None,
                "reading": "no distribution supplied -- skipped (pass "
                           "distribution and dist_params to run it)"}
    from entropy_mcp import distributions as dist_mod
    xs = sorted(values)
    n = len(xs)
    d = 0.0
    for i, x in enumerate(xs):
        f = dist_mod.cdf(distribution, dist_params or {}, x)
        d = max(d, abs((i + 1) / n - f), abs(f - i / n))
    p = ks_pvalue(d, n)
    return {"statistic": d, "p_value": p, "passed": p > alpha,
            "reading": f"KS distance {d:.4f} vs {distribution} "
                       f"{'is' if p > alpha else 'is not'} consistent"}


def _t_gap(kind, values, alpha, expected=None):
    """Gaps between recurrences of the most common symbol vs geometric."""
    counts = Counter(values)
    if len(counts) < 2:
        return {"statistic": None, "p_value": None, "passed": None,
                "reading": "single distinct value -- gap test not "
                           "applicable"}
    sym = counts.most_common(1)[0][0]
    p_sym = counts[sym] / len(values)
    gaps = []
    last = None
    for i, v in enumerate(values):
        if v == sym:
            if last is not None:
                gaps.append(i - last - 1)
            last = i
    if len(gaps) < 10:
        return {"statistic": None, "p_value": None, "passed": None,
                "reading": "too few recurrences for a gap test"}
    # Bin gaps into geometric-probability buckets and chi-square them.
    max_gap = max(gaps)
    # Expected probability of gap g is p * (1-p)^g. Bucket so each
    # expected count is >= ~5.
    buckets = {}
    cum = 0.0
    edges = [0]
    g = 0
    target = max(5.0, len(gaps) / 10.0)
    while g <= max_gap:
        cum += p_sym * (1.0 - p_sym) ** g
        if cum * len(gaps) >= target or g == max_gap:
            edges.append(g + 1)
            cum = 0.0
        g += 1
    if len(edges) < 3:
        return {"statistic": None, "p_value": None, "passed": None,
                "reading": "gap distribution too narrow to test"}
    obs = [0] * (len(edges) - 1)
    exp = [0.0] * (len(edges) - 1)
    for gap in gaps:
        for b in range(len(edges) - 1):
            if edges[b] <= gap < edges[b + 1]:
                obs[b] += 1
                break
        else:
            obs[-1] += 1
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        exp[b] = len(gaps) * sum(
            p_sym * (1.0 - p_sym) ** gg for gg in range(lo, hi))
    stat = sum((o - e) ** 2 / e for o, e in zip(obs, exp) if e > 0)
    df = sum(1 for e in exp if e > 0) - 1
    p = chi2_pvalue(stat, df) if df > 0 else None
    return {"statistic": stat, "p_value": p,
            "passed": (p is None) or p > alpha,
            "reading": f"gaps between recurrences of {sym!r} "
                       f"{'follow' if (p is None or p > alpha) else 'do not follow'} "
                       f"a geometric distribution"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def test_entropy(sequence, tests="all", alpha=0.05, input_type="auto",
                 expected=None, distribution=None, dist_params=None):
    """Run statistical tests against a sequence.

    tests: "all" or a list from ALL_TESTS. alpha: significance level.
    expected: optional {symbol: probability} for chi_squared.
    distribution/dist_params: optional, enables the KS test on numeric
    input (uses the same six names as distributions.py).
    """
    if not 0 < alpha < 1:
        raise ParameterError("alpha must be in (0, 1)",
                             parameter="alpha", expected="0 < alpha < 1")
    kind, values = _infer(sequence, input_type)
    if tests == "all":
        selected = list(ALL_TESTS)
    elif isinstance(tests, (list, tuple)):
        selected = [t for t in tests if t in ALL_TESTS]
        unknown = set(tests) - set(ALL_TESTS)
        if unknown:
            raise ParameterError(
                f"unknown tests: {sorted(unknown)}",
                parameter="tests", expected=" | ".join(ALL_TESTS))
    else:
        raise ParameterError("tests must be 'all' or a list",
                             parameter="tests",
                             expected="'all' or list of test names")

    runners = {
        "frequency": _t_frequency,
        "chi_squared": _t_chi_squared,
        "runs": _t_runs,
        "longest_run": _t_longest_run,
        "serial": _t_serial,
        "entropy": _t_entropy,
        "ks": lambda k, v, a, e: _t_ks(k, v, a, e, distribution,
                                      dist_params),
        "gap": _t_gap,
    }

    results = {}
    passed_count = 0
    run_count = 0
    for name in selected:
        r = runners[name](kind, values, alpha, expected)
        results[name] = r
        if r["passed"] is not None:
            run_count += 1
            if r["passed"]:
                passed_count += 1

    return {
        "input_type": kind,
        "n": len(values),
        "alpha": alpha,
        "tests": results,
        "tests_run": run_count,
        "tests_passed": passed_count,
        "verdict": ("pass" if run_count > 0 and passed_count == run_count
                    else "fail" if run_count > 0 else "inconclusive"),
        "note": "informed by NIST SP 800-22; no certification claimed. "
                "A battery at alpha=0.05 fails ~1 test in 20 on good "
                "entropy by chance alone.",
    }

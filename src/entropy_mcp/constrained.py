"""Attribute-constrained sampling -- the compliance wedge.

Generates records where each attribute is sampled INDEPENDENTLY from a
caller-supplied target distribution, then reports a conformance block
per attribute: target vs realized proportions, a chi-squared test, and
a verdict. The provable claim is "we targeted this distribution and
realized that one, here is the test" -- the attested artifact is the
product.

Independence is the property being sold. Each attribute consumes its
own uniform per record, in attribute insertion order; no joint
sampling, no correlation structure.

Stream consumption order: for each record (in order), for each
attribute (in the caller's dict insertion order), one uniform() draw
bisected into that attribute's cumulative weight array.
"""

import bisect
import math

from entropy_mcp.errors import ParameterError
from entropy_mcp.source import get_stream, uniform
from entropy_mcp.testsuite import chi2_pvalue

MAX_N = 10000
SUM_TOL = 1e-9


def _validate_attributes(attributes):
    if not isinstance(attributes, dict) or not attributes:
        raise ParameterError(
            "attributes must be a non-empty object mapping "
            "attribute name -> {value: proportion}",
            parameter="attributes",
            expected="{attr: {value: proportion, ...}}")
    clean = {}
    for attr, dist in attributes.items():
        if not isinstance(dist, dict) or not dist:
            raise ParameterError(
                f"attributes.{attr} must be a non-empty object of "
                f"value -> proportion",
                parameter=attr, expected="{value: proportion}")
        total = 0.0
        for val, p in dist.items():
            if not isinstance(p, (int, float)) or isinstance(p, bool) \
                    or not math.isfinite(p) or p < 0:
                raise ParameterError(
                    f"attributes.{attr}.{val} must be a non-negative "
                    f"finite number, got {p!r}",
                    parameter=attr, expected="non-negative floats")
            total += p
        if abs(total - 1.0) > SUM_TOL:
            raise ParameterError(
                f"attributes.{attr} proportions sum to {total}, "
                f"must be 1.0 within {SUM_TOL}",
                parameter=attr, expected="proportions summing to 1.0")
        clean[attr] = dict(dist)
    return clean


def sample_constrained(attributes, n, seed=None):
    """Generate n records with independently sampled attributes.

    Returns the records plus a conformance block per attribute with
    target/realized proportions, counts, chi-squared, p-value, max
    absolute deviation, and a verdict.

    Verdict thresholds: "conforms" if p > 0.05, else "deviates". At
    small n, sampling noise dominates and the test has low power -- the
    verdict describes this sample, not the generator.
    """
    clean = _validate_attributes(attributes)
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_N:
        raise ParameterError(
            f"n must be an integer between 1 and {MAX_N}",
            parameter="n", expected=f"1..{MAX_N}")

    # Precompute per-attribute cumulative arrays once.
    attr_order = list(clean.keys())
    cum = {}
    for attr in attr_order:
        prefix = []
        run = 0.0
        for val, p in clean[attr].items():
            run += p
            prefix.append((run, val))
        cum[attr] = prefix

    stream, seed_hex = get_stream(seed)

    records = []
    counts = {attr: {val: 0 for val in clean[attr]} for attr in attr_order}
    for _ in range(n):
        rec = {}
        for attr in attr_order:
            prefix = cum[attr]
            r = uniform(stream)
            i = bisect.bisect_left([p for p, _ in prefix], r)
            if i >= len(prefix):
                i = len(prefix) - 1
            val = prefix[i][1]
            rec[attr] = val
            counts[attr][val] += 1
        records.append(rec)

    conformance = {}
    for attr in attr_order:
        target = clean[attr]
        realized = {v: counts[attr][v] / n for v in target}
        stat = 0.0
        for val, p in target.items():
            e = p * n
            o = counts[attr][val]
            if e > 0:
                stat += (o - e) ** 2 / e
            elif o > 0:
                stat += o
        df = len(target) - 1
        p_value = chi2_pvalue(stat, df) if df > 0 else None
        max_dev = max(abs(realized[v] - target[v]) for v in target)
        verdict = ("conforms" if (p_value is None or p_value > 0.05)
                   else "deviates")
        conformance[attr] = {
            "target": target,
            "realized": realized,
            "counts": counts[attr],
            "chi_squared": stat,
            "p_value": p_value,
            "max_abs_deviation": max_dev,
            "verdict": verdict,
        }

    result = {
        "n": n,
        "attributes": attr_order,
        "records": records,
        "conformance": conformance,
        "seeded": stream.seeded,
        "note": "attributes are sampled independently per record; at "
                "small n the chi-squared test has low power and noise "
                "dominates -- 'conforms' describes this sample.",
    }
    if seed_hex is not None:
        result["seed"] = seed_hex
    return result

"""Cross-run reproducibility: fixed seeds must produce byte-exact output.

These golden fixtures are a compatibility contract. If a code change
alters any of them, that is a breaking change requiring a version bump
-- do not edit the fixtures to match new output.

Also verified: the same seed in a separate Python process yields
identical output (subprocess check).
"""

import json
import os
import subprocess
import sys

import pytest

from entropy_mcp import source
from entropy_mcp.distributions import sample

SEED = "0123456789abcdef" * 4  # 64 hex chars


# ---------------------------------------------------------------------------
# Golden fixtures -- frozen, do not regenerate casually.
# ---------------------------------------------------------------------------

def test_random_int_golden():
    r = source.random_int(0, 99, count=10, seed=SEED)
    assert r["values"] == GOLDEN_INTS


def test_shuffle_golden():
    r = source.shuffle(["a", "b", "c", "d", "e", "f"], seed=SEED)
    assert r["shuffled"] == GOLDEN_SHUFFLE
    assert r["permutation"] == GOLDEN_PERM


def test_choose_golden():
    r = source.choose(["p", "q", "r", "s"], k=2,
                      weights=[0.1, 0.2, 0.3, 0.4], seed=SEED)
    assert r["indices"] == GOLDEN_CHOOSE_IDX


def test_sample_normal_golden():
    r = sample("normal", {"mean": 0, "sd": 1}, n=4, seed=SEED)
    assert r["samples"] == GOLDEN_NORMAL


def test_sample_beta_golden():
    r = sample("beta", {"alpha": 2, "beta": 5}, n=3, seed=SEED)
    assert r["samples"] == GOLDEN_BETA


def test_stream_bytes_golden():
    s = source.DeterministicStream(bytes.fromhex(SEED))
    assert s.read(32).hex() == GOLDEN_STREAM32


# ---------------------------------------------------------------------------
# Cross-process determinism: a fresh interpreter must agree.
# ---------------------------------------------------------------------------

def _run_in_subprocess(script):
    env = dict(os.environ)
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    env["PYTHONPATH"] = os.path.abspath(src)
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_cross_process_random_int():
    script = (
        "import json; from entropy_mcp import source; "
        f"print(json.dumps(source.random_int(0, 99, count=10, "
        f"seed='{SEED}')['values']))")
    assert _run_in_subprocess(script) == GOLDEN_INTS


def test_cross_process_shuffle():
    script = (
        "import json; from entropy_mcp import source; "
        f"print(json.dumps(source.shuffle(['a','b','c','d','e','f'], "
        f"seed='{SEED}')['shuffled']))")
    assert _run_in_subprocess(script) == GOLDEN_SHUFFLE


def test_cross_process_sample():
    script = (
        "import json; from entropy_mcp.distributions import sample; "
        f"print(json.dumps(sample('normal', {{'mean':0,'sd':1}}, n=4, "
        f"seed='{SEED}')['samples']))")
    assert _run_in_subprocess(script) == GOLDEN_NORMAL


# ---------------------------------------------------------------------------
# Modulo bias regression: the reason rejection sampling exists.
# ---------------------------------------------------------------------------

def _modulo_biased_ints(n, lo, hi):
    """The WRONG way: os.urandom bytes mod range. Kept here to prove it
    fails the uniformity check our rejection sampler passes."""
    span = hi - lo + 1
    return [lo + b % span for b in os.urandom(n)]


def _chi2_uniform(counts):
    n = sum(counts)
    k = len(counts)
    e = n / k
    stat = sum((c - e) ** 2 / e for c in counts)
    from entropy_mcp.testsuite import chi2_pvalue
    return chi2_pvalue(stat, k - 1)


def test_rejection_sampler_passes_uniformity():
    """1,000,000 draws in [0,9] via the rejection path: chi2 p > 0.01.

    Drives randbelow on an OSStream directly -- the same code path
    random_int uses, without the per-call count cap."""
    stream = source.OSStream()
    counts = [0] * 10
    for _ in range(1_000_000):
        counts[source.randbelow(stream, 10)] += 1
    assert _chi2_uniform(counts) > 0.01


def test_modulo_implementation_fails_uniformity():
    """The deliberately biased version must FAIL the same check.

    os.urandom gives bytes 0..255; 256 mod 10 leaves 0..5 over-
    represented by ~2%. Over 1M draws that bias is glaring.
    """
    vals = _modulo_biased_ints(1_000_000, 0, 9)
    counts = [vals.count(i) for i in range(10)]
    assert _chi2_uniform(counts) <= 0.01


# ---------------------------------------------------------------------------
# Golden values (generated once, then frozen). Compatibility contract:
# a change here means a breaking version bump, not a fixture edit.
# ---------------------------------------------------------------------------

GOLDEN_STREAM32 = ("95eeed9a15059919dc31d78813481c11cca743e8a4fbe6c75c34c56c"
                   "253596ee")
GOLDEN_INTS = [21, 26, 21, 5, 25, 25, 92, 49, 87, 8]
GOLDEN_SHUFFLE = ["a", "e", "d", "b", "c", "f"]
GOLDEN_PERM = [0, 4, 3, 1, 2, 5]
GOLDEN_CHOOSE_IDX = [2, 1]
GOLDEN_NORMAL = [0.6600268397420685, -0.7964535322727154,
                 -0.4270956719066208, 0.5150812934112038]
GOLDEN_BETA = [0.3314499134704172, 0.4853129625690309,
               0.14363993012970425]

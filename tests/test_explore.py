"""Tests for bandit strategies."""

import pytest

from entropy_mcp import explore
from entropy_mcp.errors import ParameterError

SEED = "b0a7" * 16

ARMS = ["a", "b", "c"]
STATE = {
    "a": {"pulls": 10, "successes": 7, "total_reward": 7.0},
    "b": {"pulls": 10, "successes": 3, "total_reward": 3.0},
    "c": {"pulls": 0, "successes": 0, "total_reward": 0.0},
}


def test_epsilon_greedy_exploit():
    r = explore.explore(ARMS, "epsilon_greedy", STATE,
                        {"epsilon": 0.0}, seed=SEED)
    assert r["arm"] == "a"  # highest mean
    assert r["reason"] == "exploit"


def test_epsilon_greedy_explore():
    # epsilon=1.0 always explores; over seeds it should hit every arm
    seen = set()
    for i in range(60):
        r = explore.explore(ARMS, "epsilon_greedy", STATE,
                            {"epsilon": 1.0}, seed=f"{i:064x}")
        assert r["reason"] == "explore"
        seen.add(r["arm"])
    assert seen == set(ARMS)


def test_epsilon_greedy_tie_break_is_random():
    """Tied arms must be broken randomly, not by index order."""
    state = {a: {"pulls": 5, "successes": 2, "total_reward": 2.0}
             for a in ARMS}
    seen = set()
    for i in range(90):
        r = explore.explore(ARMS, "epsilon_greedy", state,
                            {"epsilon": 0.0}, seed=f"{i:064x}")
        seen.add(r["arm"])
    assert len(seen) > 1  # not always arms[0]


def test_thompson_picks_posterior_leader():
    r = explore.explore(ARMS, "thompson", STATE, seed=SEED)
    assert r["arm"] in ARMS
    assert set(r["scores"]) == set(ARMS)
    # with these posteriors, 'a' should win most of the time
    # (random baseline is ~33; true rate ~75%)
    wins = sum(
        explore.explore(ARMS, "thompson", STATE, seed=f"{i:064x}")["arm"]
        == "a" for i in range(100))
    assert wins > 55


def test_ucb1_selects_unpulled_first():
    r = explore.explore(ARMS, "ucb1", STATE, seed=SEED)
    assert r["arm"] == "c"
    assert r["reason"] == "explore"


def test_ucb1_exploit():
    state = {a: {"pulls": 10, "successes": 5, "total_reward": 5.0}
             for a in ARMS}
    state["b"]["total_reward"] = 9.0
    state["b"]["successes"] = 9
    r = explore.explore(ARMS, "ucb1", state, seed=SEED)
    assert r["arm"] == "b"


def test_cold_start():
    r = explore.explore(ARMS, "epsilon_greedy", {}, {"epsilon": 0.0},
                      seed=SEED)
    assert r["arm"] in ARMS


def test_seeded_reproducible():
    a = explore.explore(ARMS, "thompson", STATE, seed=SEED)
    b = explore.explore(ARMS, "thompson", STATE, seed=SEED)
    assert a["arm"] == b["arm"]
    assert a["scores"] == b["scores"]


def test_validation():
    with pytest.raises(ParameterError):
        explore.explore([], "ucb1", {})
    with pytest.raises(ParameterError):
        explore.explore(ARMS, "softmax", {})
    with pytest.raises(ParameterError):
        explore.explore(ARMS, "epsilon_greedy", {}, {"epsilon": 1.5})
    with pytest.raises(ParameterError):
        explore.explore(ARMS, "ucb1", {"a": {"pulls": -1}})

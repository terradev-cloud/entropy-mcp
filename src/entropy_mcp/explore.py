"""Bandit strategies -- small, closed-form, stateless.

The caller passes state in and gets a selection back; the caller owns
state persistence entirely. Three strategies:

    epsilon_greedy  -- with probability epsilon, pick uniformly at
                       random; otherwise pick the highest mean reward.
                       Ties are broken by a random draw, never by index
                       order -- index-order tie-breaking is itself a bias.
    thompson        -- sample Beta(successes + 1, failures + 1) per arm
                       using the beta sampler from distributions.py,
                       pick the max.
    ucb1            -- deterministic upper confidence bound; included
                       for completeness. Unpulled arms are selected
                       first (random among them); remaining ties are
                       broken randomly.

Stream consumption order: epsilon_greedy consumes one uniform for the
explore/exploit decision, then one randbelow for the pick (explore or
tie-break). thompson consumes per-arm beta draws in arm order. ucb1
consumes only when a random tie-break is needed.
"""

import math

from entropy_mcp.errors import ParameterError
from entropy_mcp.source import get_stream, randbelow, uniform

STRATEGIES = ("epsilon_greedy", "thompson", "ucb1")


def _validate(arms, strategy, state):
    if not isinstance(arms, (list, tuple)) or len(arms) < 1:
        raise ParameterError("arms must be a non-empty list",
                             parameter="arms", expected="list of arm ids")
    arms = list(arms)
    if strategy not in STRATEGIES:
        raise ParameterError(
            f"strategy must be one of {list(STRATEGIES)}, "
            f"got {strategy!r}",
            parameter="strategy", expected=" | ".join(STRATEGIES))
    if state is None:
        state = {}
    if not isinstance(state, dict):
        raise ParameterError("state must be an object mapping arm id -> "
                             "{pulls, successes, total_reward}",
                             parameter="state", expected="object")
    clean = {}
    for arm in arms:
        s = state.get(str(arm)) or state.get(arm) or {}
        pulls = s.get("pulls", 0)
        succ = s.get("successes", 0)
        rew = s.get("total_reward", 0.0)
        for name, v in (("pulls", pulls), ("successes", succ)):
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ParameterError(
                    f"state[{arm}].{name} must be a non-negative int",
                    parameter="state", expected="non-negative integers")
        if not isinstance(rew, (int, float)) or isinstance(rew, bool) \
                or not math.isfinite(rew):
            raise ParameterError(
                f"state[{arm}].total_reward must be a finite number",
                parameter="state", expected="finite number")
        clean[arm] = {"pulls": pulls, "successes": succ,
                      "total_reward": float(rew)}
    return arms, clean


def _pick_random(stream, candidates):
    """Uniform pick among candidates -- the honest tie-break."""
    return candidates[randbelow(stream, len(candidates))]


def _argmax_random(scores, stream):
    """Argmax with random tie-breaking (never index order)."""
    best = max(scores.values())
    tied = [a for a, s in scores.items() if s == best]
    return _pick_random(stream, tied), tied


def explore(arms, strategy, state=None, params=None, seed=None):
    """Select an arm under the given strategy.

    Returns the selected arm, the reason ("explore" | "exploit"), the
    per-arm scores used for the decision, and the seed used.
    """
    params = params or {}
    arms, st = _validate(arms, strategy, state)
    stream, seed_hex = get_stream(seed)

    if strategy == "epsilon_greedy":
        epsilon = params.get("epsilon", 0.1)
        if not isinstance(epsilon, (int, float)) \
                or isinstance(epsilon, bool) or not 0 <= epsilon <= 1:
            raise ParameterError(
                "epsilon must be in [0, 1]",
                parameter="epsilon", expected="0 <= epsilon <= 1")
        means = {a: (s["total_reward"] / s["pulls"] if s["pulls"] else 0.0)
                 for a, s in st.items()}
        if uniform(stream) < epsilon:
            arm = _pick_random(stream, arms)
            reason = "explore"
        else:
            arm, _ = _argmax_random(means, stream)
            reason = "exploit"
        scores = means

    elif strategy == "thompson":
        from entropy_mcp.distributions import _beta_cheng, _beta_johnk
        scores = {}
        for a in arms:
            s = st[a]
            alpha = s["successes"] + 1.0
            beta = (s["pulls"] - s["successes"]) + 1.0
            if alpha < 1.0 and beta < 1.0:
                scores[a] = _beta_johnk(stream, alpha, beta)
            else:
                scores[a] = _beta_cheng(stream, alpha, beta)
        arm, _ = _argmax_random(scores, stream)
        reason = "exploit"  # thompson explores via posterior spread

    else:  # ucb1
        total_pulls = sum(s["pulls"] for s in st.values())
        unpulled = [a for a in arms if st[a]["pulls"] == 0]
        if unpulled:
            arm = _pick_random(stream, unpulled)
            reason = "explore"
            scores = {a: (float("inf") if st[a]["pulls"] == 0 else
                          st[a]["total_reward"] / st[a]["pulls"])
                      for a in arms}
        else:
            log_t = math.log(max(total_pulls, 1))
            scores = {}
            for a in arms:
                s = st[a]
                mean = s["total_reward"] / s["pulls"]
                scores[a] = mean + math.sqrt(2.0 * log_t / s["pulls"])
            arm, _ = _argmax_random(scores, stream)
            reason = "exploit"

    # JSON can't carry inf -- flatten scores for the response.
    out_scores = {str(a): (None if s == float("inf") else s)
                  for a, s in scores.items()}
    result = {
        "arm": arm,
        "reason": reason,
        "strategy": strategy,
        "scores": out_scores,
        "seeded": stream.seeded,
    }
    if seed_hex is not None:
        result["seed"] = seed_hex
    return result

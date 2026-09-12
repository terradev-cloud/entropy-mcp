#!/usr/bin/env python3
"""entropy -- cryptographically sound randomness for agents, with proof.

Raw JSON-RPC 2.0 over stdio, no MCP library, no asyncio in the core:

    read line from stdin -> parse JSON-RPC -> dispatch -> write -> flush

Four tools, full functionality:

    random      every direct draw: bytes | int | shuffle | choose |
                sample (six named distributions) | constrained
                (attribute-constrained records with a conformance
                proof). `operation` selects the primitive, `params`
                carries its arguments, `seed` makes it reproducible.
    commitment  the verifiable chain: commit | reveal | draw |
                verify | list
    test        statistical battery informed by NIST SP 800-22
    explore     epsilon-greedy / thompson / ucb1 arm selection

Every seeded operation is deterministic given its inputs and seed; only
the raw OS-entropy paths are non-deterministic, and they say so.
"""

import json
import sys

from entropy_mcp import __version__
from entropy_mcp.errors import EntropyError, ParameterError

RANDOM_OPS = ("bytes", "int", "shuffle", "choose", "sample",
              "constrained")
COMMITMENT_OPS = ("commit", "reveal", "draw", "verify", "list")

TOOLS = [
    {
        "name": "random",
        "description": (
            "Direct randomness. operation selects the primitive:\n"
            "  bytes       -- raw CSPRNG bytes. params: {count<=4096, "
            "encoding: hex|base64}\n"
            "  int         -- unbiased integers in [min,max] via "
            "rejection sampling. params: {min, max, count<=10000}\n"
            "  shuffle     -- Fisher-Yates; permutation[i] is the "
            "original index now at position i. params: {items, "
            "return_permutation}\n"
            "  choose      -- k items, weighted or not, with/without "
            "replacement. params: {items, k, weights, replacement}\n"
            "  sample      -- named distributions: uniform, normal, "
            "lognormal, exponential, triangular, beta. params: "
            "{distribution, params, n<=10000}\n"
            "  constrained -- records with independently sampled "
            "attributes + chi-squared conformance proof. params: "
            "{attributes: {attr: {value: prop}}, n}\n"
            "seed (top level) makes any operation reproducible."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": list(RANDOM_OPS)},
                "params": {"type": "object",
                           "description": "operation-specific arguments"},
                "seed": {"type": "string",
                         "description": "optional seed for a "
                                        "reproducible draw"},
            },
            "required": ["operation"],
        },
    },
    {
        "name": "commitment",
        "description": (
            "Verifiable commit-reveal. operation selects the step:\n"
            "  commit -- hash a seed, bind it to draw_spec "
            "{operation, params}, attest via Stamp. The seed is "
            "returned to the caller and never stored.\n"
            "  reveal -- verify a seed against commitment_id, "
            "re-execute the draw deterministically, attest the "
            "outcome.\n"
            "draw   -- commit + execute + reveal in one call. Pass "
            "draw_spec, or operation + params directly.\n"
            "verify -- pure third-party verification of "
            "commitment_record + reveal_record. No local state.\n"
            "list   -- read-only view of the local commitment log "
            "(limit, since)."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string",
                              "enum": list(COMMITMENT_OPS)},
                "draw_spec": {"type": "object",
                              "description": "{operation, params} for "
                                             "commit/draw"},
                "commitment_id": {"type": "string"},
                "seed": {"type": "string"},
                "description": {"type": "string", "default": ""},
                "attested": {"type": "boolean", "default": True},
                "anchor_beacon": {"type": "boolean", "default": False},
                "commitment_record": {"type": "object"},
                "reveal_record": {"type": "object"},
                "limit": {"type": "integer", "default": 50,
                          "minimum": 1, "maximum": 1000},
                "since": {"type": "string",
                          "description": "ISO 8601 lower bound"},
            },
            "required": ["operation"],
        },
    },
    {
        "name": "test",
        "description": (
            "Statistical battery over a sequence: frequency, "
            "chi-squared, runs, longest run, serial correlation, "
            "Shannon entropy, KS, gap test. Informed by NIST SP 800-22; "
            "no certification claimed. sequence is a list of numbers, "
            "list of strings, or a hex/base64 byte string."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sequence": {"description": "list of numbers, list of "
                                            "strings, or hex/base64"},
                "tests": {"default": "all",
                          "description": "'all' or a list of: frequency, "
                                         "chi_squared, runs, "
                                         "longest_run, serial, entropy, "
                                         "ks, gap"},
                "alpha": {"type": "number", "default": 0.05},
                "input_type": {"type": "string",
                               "enum": ["auto", "numeric", "categorical",
                                        "bytes"], "default": "auto"},
                "expected": {"type": "object",
                             "description": "{symbol: probability} for "
                                            "chi_squared"},
                "distribution": {"type": "string",
                                 "description": "enables the KS test"},
                "dist_params": {"type": "object"},
            },
            "required": ["sequence"],
        },
    },
    {
        "name": "explore",
        "description": (
            "Bandit arm selection: epsilon_greedy, thompson, or ucb1. "
            "Stateless -- caller passes state in as {arm: {pulls, "
            "successes, total_reward}} and gets a selection back. "
            "params: {epsilon} for epsilon_greedy."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "arms": {"type": "array"},
                "strategy": {"type": "string",
                             "enum": ["epsilon_greedy", "thompson",
                                      "ucb1"]},
                "state": {"type": "object"},
                "params": {"type": "object"},
                "seed": {"type": "string"},
            },
            "required": ["arms", "strategy"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def call_tool(name, arguments):
    """Execute one tool. Returns an MCP content block dict."""
    arguments = arguments or {}
    try:
        result = _run_tool(name, arguments)
    except EntropyError as exc:
        return {"content": [{"type": "text",
                             "text": json.dumps(exc.to_dict())}],
                "isError": True}
    except Exception as exc:
        err = {"error": {"code": "internal_error",
                         "message": f"{type(exc).__name__}: {exc}",
                         "parameter": None, "expected": None}}
        return {"content": [{"type": "text", "text": json.dumps(err)}],
                "isError": True}
    return {"content": [{"type": "text", "text": json.dumps(result)}]}


def _run_tool(name, a):
    if name == "random":
        return _random_op(a)
    if name == "commitment":
        return _commitment_op(a)
    if name == "test":
        from entropy_mcp import testsuite
        return testsuite.test_entropy(
            a.get("sequence"), a.get("tests", "all"), a.get("alpha", 0.05),
            a.get("input_type", "auto"), a.get("expected"),
            a.get("distribution"), a.get("dist_params"))
    if name == "explore":
        from entropy_mcp import explore
        return explore.explore(a.get("arms"), a.get("strategy"),
                               a.get("state"), a.get("params"),
                               a.get("seed"))
    raise EntropyError(f"unknown tool: {name}")


def _random_op(a):
    """Dispatch a `random` call: operation + params + optional seed."""
    from entropy_mcp import constrained, distributions, source

    op = a.get("operation")
    if op not in RANDOM_OPS:
        raise ParameterError(
            f"operation must be one of {list(RANDOM_OPS)}, got {op!r}",
            parameter="operation", expected=" | ".join(RANDOM_OPS))
    p = a.get("params") or {}
    if not isinstance(p, dict):
        raise ParameterError("params must be an object",
                             parameter="params", expected="object")
    seed = a.get("seed")

    if op == "bytes":
        return source.random_bytes(p.get("count", 32),
                                   p.get("encoding", "hex"))
    if op == "int":
        return source.random_int(p.get("min"), p.get("max"),
                                 p.get("count", 1), seed)
    if op == "shuffle":
        return source.shuffle(p.get("items"), seed,
                              p.get("return_permutation", True))
    if op == "choose":
        return source.choose(p.get("items"), p.get("k", 1),
                             p.get("weights"), p.get("replacement", False),
                             seed)
    if op == "sample":
        result = distributions.sample(p.get("distribution"),
                                      p.get("params", {}),
                                      p.get("n", 1), seed)
        from entropy_mcp import stamp_client
        result["params_canonical_hash"] = stamp_client.canonical_hash(
            result["params"])
        return result
    # constrained
    return constrained.sample_constrained(p.get("attributes"),
                                          p.get("n"), seed)


def _commitment_op(a):
    """Dispatch a `commitment` call across the commit-reveal chain."""
    from entropy_mcp import commit as commit_mod

    op = a.get("operation")
    if op not in COMMITMENT_OPS:
        raise ParameterError(
            f"operation must be one of {list(COMMITMENT_OPS)}, "
            f"got {op!r}",
            parameter="operation", expected=" | ".join(COMMITMENT_OPS))

    if op == "commit":
        return commit_mod.commit(a.get("draw_spec"),
                                 a.get("description", ""),
                                 a.get("seed"),
                                 a.get("anchor_beacon", False))
    if op == "reveal":
        return commit_mod.reveal(a.get("commitment_id"), a.get("seed"))
    if op == "draw":
        # Accept draw_spec {operation, params} or flat operation+params.
        spec = a.get("draw_spec")
        if isinstance(spec, dict) and "operation" in spec:
            operation = spec["operation"]
            params = spec.get("params", {})
        else:
            operation = a.get("draw_operation") or a.get("op")
            params = a.get("params", {})
        return commit_mod.draw(operation, params,
                               a.get("description", ""),
                               a.get("attested", True),
                               a.get("anchor_beacon", False),
                               a.get("seed"))
    if op == "verify":
        return commit_mod.verify_chain(a.get("commitment_record"),
                                       a.get("reveal_record"))
    # list
    return commit_mod.list_commitments(a.get("limit", 50),
                                       a.get("since"))


# ---------------------------------------------------------------------------
# JSON-RPC plumbing -- same shape as stamp-mcp's server.
# ---------------------------------------------------------------------------

def _write(message):
    """Write one JSON-RPC message as a single line, then FLUSH."""
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def error_response(msg_id, code, message):
    """Build a JSON-RPC error response object (transport-agnostic)."""
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": code, "message": message}}


def dispatch(req):
    """Dispatch one parsed JSON-RPC message. Synchronous -- the core has
    no asyncio. Returns the response dict, or None for notifications.
    Shared by the stdio transport (main) and the HTTP transport
    (entropy_mcp.http)."""
    method = req.get("method")
    msg_id = req.get("id")  # None -> notification -> never respond

    if method == "initialize":
        requested = (req.get("params") or {}).get("protocolVersion")
        result = {
            "protocolVersion": requested or "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "entropy", "version": __version__},
        }
    elif method == "notifications/initialized":
        return None
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "resources/list":
        result = {"resources": []}
    elif method == "resources/templates/list":
        result = {"resourceTemplates": []}
    elif method == "prompts/list":
        result = {"prompts": []}
    elif method == "tools/call":
        params = req.get("params") or {}
        result = call_tool(params.get("name"), params.get("arguments"))
    else:
        if msg_id is None:
            return None
        return error_response(msg_id, -32601,
                              f"Method not found: {method}")

    if msg_id is None:
        return None
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def main():
    # stdout is the protocol channel -- log to stderr only.
    print("entropy: listening on stdin", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _write(error_response(None, -32700, "Parse error"))
            continue
        try:
            resp = dispatch(req)
        except Exception as e:
            _write(error_response(req.get("id"), -32603,
                                  f"{type(e).__name__}: {e}"))
            continue
        if resp is not None:
            _write(resp)


if __name__ == "__main__":
    main()

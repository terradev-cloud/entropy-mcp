# entropy-mcp

Entropy gives agents randomness they cannot produce themselves, and proof they did not manipulate it.

## The evidence

Language models cannot generate statistically sound randomness. This is measured, not speculative:

- **[arXiv:2601.05414](https://arxiv.org/abs/2601.05414)** — 11 frontier models, 15 distributions: 7% median pass rate in batch mode; 10 of 11 models passed *zero* distributions in stateless mode.
- **[arXiv:2604.06543](https://arxiv.org/abs/2604.06543)** — the knowing-doing gap: models can describe correct sampling and still fail to do it. Asking a model to generate its own seed reproduces the same bias.
- **[arXiv:2606.06622](https://arxiv.org/abs/2606.06622)** — UnpredictaBench: models are measurably predictable where they should be random.

## The stateless point

Agents operate in exactly the regime where models fail hardest: independent, stateless calls with no memory of prior draws. Every "pick a random number" an agent answers from its own weights is a draw from a biased, predictable source. Entropy moves the draw out of the model and into the OS CSPRNG — and the commit-reveal chain proves the draw that happened is the draw that was reported.

## Install

```bash
pip install entropy-mcp          # stdio core + stamp-mcp dependency
pip install entropy-mcp[http]    # adds the aiohttp HTTP transport
```

## Quickstart

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"commitment","arguments":{
  "operation":"draw",
  "draw_spec":{"operation":"shuffle","params":{"items":["alice","bob","carol","dave"]}},
  "description":"who goes first"}}}
```

One call returns the shuffled list **plus** a commitment record, a reveal record, and a verification block — the full chain a third party can check with `commitment` `operation:"verify"`.

## Command reference

Four tools. `operation` selects the primitive; `params` carries its arguments; `seed` makes any draw reproducible.

| Tool | Operations |
|---|---|
| `random` | `bytes` (CSPRNG, hex/base64) · `int` (unbiased [min,max] via rejection sampling) · `shuffle` (Fisher-Yates + permutation) · `choose` (weighted/unweighted, with/without replacement) · `sample` (uniform, normal, lognormal, exponential, triangular, beta) · `constrained` (independent per-attribute sampling + chi² conformance proof) |
| `commitment` | `commit` (hash seed, bind draw spec, attest) · `reveal` (verify seed, re-execute, attest outcome) · `draw` (commit + execute + reveal in one call) · `verify` (pure third-party verification, no local state) · `list` (local commitment log) |
| `test` | Statistical battery: frequency, chi², runs, longest run, serial, Shannon, KS, gap |
| `explore` | Bandit selection: `epsilon_greedy`, `thompson`, `ucb1` (stateless) |

Every seeded command is deterministic given its inputs and seed — byte-identical across processes, machines, and Python versions. Only the raw OS-entropy paths are non-deterministic, and they say so.

## How commit-reveal works

1. **commit** generates (or accepts) a 32-byte seed, stores only `sha256(seed)`, binds it to your draw spec via canonical hash, timestamps it via Stamp's NTP, optionally anchors it to a drand round, and attests the whole record. The seed goes to you. It is never stored server-side.
2. **reveal** takes the seed back, verifies it hashes to the committed `seed_hash`, re-executes the draw deterministically, and attests the outcome.
3. **verify_chain** checks, independently of any server state:

| Check | Proves |
|---|---|
| `seed_produces_commitment_hash` | the revealed seed is the one committed to |
| `commitment_attestation_valid` | the commitment record is intact and attested |
| `reveal_attestation_valid` | the reveal record is intact and attested |
| `commitment_precedes_reveal` | the commit happened before the reveal |
| `outcome_reproducible` | the reported outcome is what the seed actually produces |
| `params_hash_matches` | the draw spec wasn't altered after commitment |
| `beacon_anchor_valid` | the commitment predates the drand round's publication (when anchored) |

## What this is not

- Not a statistics package — the summary block and test battery are conveniences, not an analysis suite.
- Not a simulation engine — no Monte Carlo, no sensitivity analysis, no fitting.
- Not a randomness beacon — drand anchoring is optional hardening, not a source.
- Not NIST-certified — the test battery is *informed by* SP 800-22, in the spirit of it.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ENTROPY_COMMITMENT_LOG` | `~/.entropy/commitments.jsonl` | commitment log path (0600, capped at 10k records) |
| `ENTROPY_STAMP_URL` | `https://stamp-mcp.terradev.cloud` | remote Stamp endpoint (local `stamp_mcp` import preferred) |
| `ENTROPY_BEACON_URL` | `https://api.drand.sh` | drand API base |
| `ENTROPY_HOST` / `ENTROPY_PORT` | `127.0.0.1` / `8001` | HTTP transport bind |

## Threat model

See [THREAT_MODEL.md](THREAT_MODEL.md). The honest summary: Entropy proves a draw was not manipulated; it cannot prove a draw was not discarded.

## License

Apache 2.0.

# Threat Model — entropy-mcp

Entropy's security claim is narrow and precise: **a committed draw cannot be
manipulated after commitment, and a third party can verify this without
trusting the server that issued the records.** Everything below follows from
that claim and its limits.

## What the construction proves

A commitment binds `sha256(seed)` to a canonically-hashed draw spec, an
NTP-verified timestamp, and a Stamp attestation — before the outcome exists.
A reveal discloses the seed; anyone can re-execute the deterministic draw and
confirm the outcome is what the seed produces. The commit precedes the
reveal, so the outcome could not have been chosen after the fact.

## Seed disclosure timing

The seed is the secret. Between commit and reveal, whoever holds the seed
controls whether the draw can be reproduced.

- The server generates the seed from `os.urandom` and returns it to the
  caller. **It is never persisted** — not in the commitment record, not in
  the log, not in the attestation payload. Only `sha256(seed)` is stored.
- A caller-supplied seed is accepted (multi-party protocols need this) but
  never echoed into stored state.
- If the caller loses the seed, the commitment can never be revealed — it
  remains a permanent, unredeemed promise. That is the correct failure mode.
- If the seed leaks before reveal, a holder can predict the outcome early but
  cannot change it. Prediction without manipulation is the designed property.

## Last-revealer and multi-party considerations

In a multi-party draw where each party commits a seed and the outcome combines
them (e.g., XOR), the last party to reveal learns the outcome before deciding
whether to reveal. A party that can abort silently can discard unfavorable
outcomes.

Entropy's single-seed commit-reveal does not, by itself, solve last-revealer
advantage. Mitigations are protocol-level: require all commitments to be
published to an external registry before any reveal, and treat a missing
reveal as a forfeit rather than a re-draw.

## Clock manipulation

`committed_at` and `revealed_at` come from Stamp's NTP query, not the local
clock — a caller who controls the host clock cannot backdate a commitment.
When NTP is unreachable the timestamp falls back to the local clock and the
record discloses this (`ntp_server: null`).

`anchor_beacon` hardens this further: the commitment embeds a drand round
whose randomness was published at a publicly known time, giving an
externally-verifiable lower bound on the commit time that no amount of local
clock manipulation can forge.

## Log tampering

The commitment log (`~/.entropy/commitments.jsonl`) is append-only, mode
0600, capped at 10,000 records. It is a convenience for `reveal` and
`list_commitments` — **it is not the trust root**. Every record in it carries
a Stamp attestation; a modified record fails `verify_chain`. And
`verify_chain` accepts records from any source, so a verifier never needs to
touch the log at all.

## What Entropy does NOT defend against

Stated plainly, because overclaiming here is the failure mode:

- **A caller who simply does not use it.** An agent that draws from its own
  weights produces no record and no proof. Entropy cannot force adoption.
- **A caller who discards an unfavorable revealed outcome without publishing
  it.** Entropy proves a draw was not manipulated; it cannot prove a draw was
  not discarded. If an agent commits, reveals, dislikes the result, and walks
  away, the records exist but nothing forces their publication. The
  mitigation is publishing commitments to an external, append-only registry
  *before* the draw executes — at that point a missing reveal is itself
  evidence. That registry is the paid tier's reason to exist.
- **Compromise of the host between commit and reveal.** An attacker with the
  seed and write access to the caller's records can do anything the caller
  can. Entropy assumes the caller's environment is honest about which records
  it publishes.
- **Stamp compromise.** Attestation and NTP time both come from Stamp. A
  compromised Stamp can forge timestamps and attestations. The drand anchor
  bounds this for commit time; nothing bounds it for attestation integrity.

## Explicit non-claims

- No NIST certification. The test battery is informed by SP 800-22, in the
  spirit of it.
- No formal cryptographic audit. The seeded stream is a SHA-256 counter-mode
  DRBG — fully specified and reproducible, not a novelty, but not audited.
- No defense in depth beyond the commit-reveal construction. If your threat
  model includes a hostile verifier or a hostile host, you need more than
  this tool.

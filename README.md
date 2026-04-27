# eerful

Reference implementation of **Enhanced Evaluation Receipts** (EER) — a portable,
third-party-verifiable artifact for recording the result of an LLM evaluation
executed inside a Trusted Execution Environment.

An EER binds a public evaluator definition (content-addressed), an optional
private input commitment, an attested model execution (hardware-rooted
signature), and an optional structured score. Verification is offline: a third
party with only the receipt and content-addressed storage can confirm
authenticity without trusting the producer or the inference provider's liveness.

The protocol is **EER**. The reference implementation is **eerful**.

- Spec: [`docs/spec.md`](docs/spec.md) (v0.4)
- Built on: 0G Compute Network (TeeML) + 0G Storage
- Status: in development for hackathon submission Sunday morning

## What's in the box

- `eerful` Python package — produces and verifies EERs per the spec
- `eerful.zg` — 0G TeeML inference + 0G Storage adapters
- `eerful.jig` — adapters that let any [jig](https://github.com/rankonelabs/jig)
  agent produce EERs as part of its evaluation flow
- `eerful` CLI — `verify`, `publish-evaluator`, `evaluate`
- Two example evaluators with runnable demos:
  - Trading-strategy critic (three-round iteration with chained receipts)
  - Prompt evaluation (generality proof)

## Install

```bash
uv sync
```

Requires Python 3.12 and a 0G Galileo testnet wallet with faucet funds for
real TeeML calls and Storage uploads.

## Quick verify

```bash
eerful verify path/to/receipt.json
```

The verifier fetches the evaluator bundle and attestation report from 0G
Storage, runs the seven-step verification algorithm (spec §7), and prints a
verdict plus the structured score block.

## What an EER proves (and doesn't)

A valid EER proves a specific response was produced by attested TEE hardware
under publicly defined evaluation criteria. It does **not** prove the
evaluation is correct, the score is meaningful, the model the TEE loaded
matches the model the evaluator declared (see spec §8), or that the producer
chose representative input.

Authenticity is what EER provides; correctness, completeness, and good-faith
use are higher-layer concerns. The spec is honest about this — read §2 and §8
before relying on receipts.

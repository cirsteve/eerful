# eerful

> Run your agents nuts, then make them prove it before the executor sees the trade.

Autonomous agents in 2026 will get compromised. Prompt injection through
retrieved documents, tool poisoning via malicious MCP servers,
adversarial RAG content, supply-chain attacks on cheap-model weights —
these aren't hypothetical, they're the dominant attack surface for any
agent doing real work. The defensive architecture has to assume the
cheap exploration layer is compromisable and place an **attested gate**
before any compromise can have consequences.

eerful is that gate. Run generation nuts — local models, retrieval,
spray-and-pray search, parallel sampling, untrusted tooling, whatever.
Then put a hard, attested gate before anything reaches the executor.
The gate runs an LLM evaluation in a TEE against criteria the principal
pinned by hash *before the agent ran*. Receipts prove the gate ran
honestly. The executor refuses to act without them.

## The pattern

Domain-agnostic. Trading is the headline demo because the consequences
are vivid (real money), but the architecture supports any consequential
agent action: PR-merge gating, RFC compliance, plan-vs-action validation,
fiduciary mandate compliance.

1. **Cheap exploration layer.** Anything goes — local models, retrieval,
   parallel sampling, parameter sweeps, synthesis via untrusted
   tooling. Optimize for breadth, speed, creativity. Don't bother
   attesting it.
2. **Pinned criteria.** A principal commits (by content hash) to one
   or more evaluator bundles before the agent runs. The bundles
   encode what "acceptable" means in this domain.
3. **Attested gate.** Before any consequential action, candidate
   outputs run through the bundle(s) inside a TEE. Each call produces
   a receipt — a portable, third-party-verifiable record of the
   evaluation, signed by an enclave-born key.
4. **Hard refuse.** The executor checks receipts against the
   committed bundle hashes and the principal's tier policy. No
   matching valid receipts → no action. Period.

eerful is the implementation of (3) and (4). The exploration layer is
the user's; (2) is the principal's job.

The diversity rule (high-consequence tier: N receipts from N distinct
TEE signers) extends the threat model — an attacker now has to compromise
the cheap exploration layer *and* N independent TEEs for a bad action
to execute. Defense in depth at the architectural level.

## What's in the box

- **`eerful` Python package** — produces and verifies EERs per the
  EER spec.
- **`eerful.policy`** — `PrincipalPolicy` schema: bundle hash
  registry + per-tier attestation requirements (N, score threshold,
  category constraints, diversity rules).
- **`eerful.executor.evaluate_gate`** — the six-check gate. Reusable
  from any orchestrator; the CLI is a thin wrapper.
- **`eerful.commitment`** — input-commitment construction for the
  chain pattern.
- **`eerful.zg`** — 0G TeeML inference + 0G Storage adapters.
- **`eerful.jig`** — adapters that let any
  [jig](https://github.com/rankonelabs/jig) agent produce EERs as
  part of its evaluation flow.
- **`eerful` CLI** — `verify`, `publish-evaluator`, `gate`.
- **`services/zg-bridge/`** — TypeScript HTTP service that wraps the
  0G broker + storage SDKs (the Python adapters drive it locally over
  loopback). See [`services/zg-bridge/README.md`](services/zg-bridge/README.md).
- **`examples/trading/`** — the headline demo: a two-gate trading
  agent (`proposal_grade` + `implementation_grade`) with a recordable
  happy path and a recordable poisoned path that the rails refuse on
  screen. See [`examples/trading/README.md`](examples/trading/README.md).

## Install

```bash
uv sync
```

Requires Python 3.12. For producing receipts (and for the default
verify / gate paths, which fetch artifacts from 0G Storage), also
start the bridge — see
[`services/zg-bridge/README.md`](services/zg-bridge/README.md) for
`npm install` + env vars. Fully-offline verification with `--bundle`,
`--report`, and `--skip-step-5` flags can avoid the bridge entirely.

The bridge listens on `127.0.0.1:7878` by default; the `eerful` CLI
refuses non-loopback bridge URLs unless you pass `--allow-remote-bridge`.
Library callers using `BridgeStorageClient` / `ComputeClient` directly
accept any URL — the loopback guard is CLI-only.

## Quick gate

```bash
eerful gate \
    --policy examples/trading/principal_policy.json \
    --tier low_consequence \
    --bundle proposal_grade \
    --receipt path/to/proposal_receipt.json
```

Exits `0` on PASS, `1` on REFUSE (with the failing check named in
`outcome`), `2` on a wiring error (unknown tier, missing file, etc.).
Stdout/stderr formatting:

```
PASS — 1 receipt(s) under bundle 'proposal_grade', tier 'low_consequence'
  canonical_set_hash: 0x9d3f...
  receipts: 1 supplied / 1 required
```

```
REFUSE (refuse_score) — receipt 0xab12... overall 0.25 < tier 'low_consequence' threshold 0.7
  receipts: 1 supplied / 1 required
```

The six refusal outcomes (`refuse_insufficient_receipts`,
`refuse_bundle_mismatch`, `refuse_invalid_receipt`, `refuse_diversity`,
`refuse_category`, `refuse_score`) are machine-greppable enum values,
each named for the check that fired first.

## Quick verify

```bash
eerful verify path/to/receipt.json
```

The verifier fetches the evaluator bundle and attestation report from
0G Storage by content hash (via the local bridge), runs the §7.1
verification algorithm, and prints a verdict plus the structured
score block. `--bundle <path>` and `--report <path>` override
individual artifacts with local files; `--skip-step-5` skips the
report fetch entirely.

## What an EER proves (and doesn't)

A valid EER proves a specific response was produced by attested TEE
hardware under publicly defined evaluation criteria, with the response
signed by an enclave-born key. Authenticity is what EER provides.

It does **not** prove the evaluation is correct, the score is
meaningful, the producer chose representative input, or — most
importantly for honest framing — that the model the TEE loaded matches
the model the evaluator declared.

That last gap is the subject of [spec §8](docs/spec.md#8-compose-vs-model-identity-binding),
which lays out the empirical state of 0G TeeML providers as a
three-tier spectrum:

- **Category A — bound launch string.** RTMR3 binds a compose whose
  launch command names the model identifier. The strongest model
  claim available today on widely deployed primitives (1 of 7
  observed mainnet providers). Weights themselves still come from
  HuggingFace at runtime and are not measured.
- **Category B — unrelated compose.** RTMR3 binds a compose that
  does not reference the advertised model at all. Receipts under such
  providers verify "some dstack TD ran some app" — nothing about the
  model.
- **Category C — centralized passthrough.** The compose attests a
  broker proxy that routes to a centralized backend with no TEE
  attestation. The provider's own broker code admits this in its
  error messages.

EER's protocol-level mitigation is the **`accepted_compose_hashes`**
allowlist (spec §6.5): an evaluator publisher pins the compose-hashes
of providers whose configuration they have inspected, *along with the
publisher's declared category*. Verification Step 5 fails closed when
the attested hash isn't in the allowlist. The executor's policy can
further constrain to a category subset (`required_categories: ["A"]`
for high-consequence) — receipts from Category B/C providers refuse
even if they're allowlisted.

`eerful verify` surfaces the §8 category and gating status (`enforced`
/ `skipped`) on stdout by default. Read §2 and §8 before relying on
receipts.

## Substrate

This implementation runs against 0G's TeeML / Storage primitives on
the Galileo testnet — chosen because the receipt-attestation primitives
are observable and the cost is faucet-cheap. The architecture is
substrate-independent: the executor reads receipts and policies, not
substrate. If 0G's ecosystem matures (per §6 of the design doc, the
high-consequence-tier supply gap on Cat A is real today), the rails
already enforce against it. If the right substrate turns out to be
Phala, Marlin, AWS Nitro, or something not yet shipped, the executor
ports without changes.

## Producing receipts from a jig pipeline

`eerful.jig` plugs into [jig](https://github.com/rankonelabs/jig)'s
`LLMClient` / `Grader` / `TracingLogger` interfaces so any pipeline
step graded by an `EvaluationGrader` produces a verifying receipt as
a side effect.

```python
import asyncio

from eerful.jig import EvaluationClient, EvaluationGrader
from eerful.evaluator import EvaluatorBundle
from eerful.zg.compute import ComputeClient
from eerful.zg.storage import BridgeStorageClient
from jig.core.pipeline import PipelineConfig, Step, run_pipeline
from jig.tracing.sqlite import SQLiteTracer


async def my_step(ctx: dict) -> str:
    """Whatever your pipeline step does. The grader scores the
    `(ctx['input'], <return value>)` pair against the bundle's prompt."""
    return ctx["input"]


async def main():
    bundle = EvaluatorBundle.model_validate_json(open("bundle.json").read())

    with (
        ComputeClient(bridge_url="http://127.0.0.1:7878") as compute,
        BridgeStorageClient(bridge_url="http://127.0.0.1:7878") as storage,
    ):
        client = EvaluationClient(
            compute=compute,
            storage=storage,
            bundle=bundle,
            evaluator_id=bundle.evaluator_id(),
            provider_address="0x...",
        )
        grader = EvaluationGrader(client=client)

        config = PipelineConfig(
            name="trading-proposal-grade",
            steps=[Step(name="grader", fn=my_step, grader=grader)],
            tracer=SQLiteTracer("traces.db"),
        )
        result = await run_pipeline(config, input="strategy-v1")


asyncio.run(main())

# Each grader call produced an EER. The receipt is on the LLM_CALL span's
# metadata under `eerful.receipt_id`, and persisted to FeedbackLoop (when
# one is configured on the grader). Subsequent calls auto-chain via
# `previous_receipt_id`; pin per-call with provider_params.
```

The grader only emits scores for top-level numeric fields of the
bundle's `output_schema`; `score_dimensions=[...]` filters further.
Tools, mismatched system prompts, and conflicting inference params
raise `EvaluationClientError` — the receipt has to attest the
bundle's criteria, not the caller's.

## Spec

- [`docs/spec.md`](docs/spec.md) — EER protocol (v0.5)
- [`specs/executor-and-rails-design.md`](specs/executor-and-rails-design.md) — executor + multi-attestation rails design (gitignored; local artifact)

Status: pre-hackathon submission. The trading demo
([`examples/trading/`](examples/trading/)) is the recordable
end-to-end artifact; everything else is the framework that lets you
build your own.

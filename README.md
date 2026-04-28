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
- `eerful.commitment` — §6.7 input-commitment construction +
  producer-side `SaltStore` for the chain pattern
- `eerful.zg` — 0G TeeML inference + 0G Storage adapters
- `eerful.jig` — adapters that let any [jig](https://github.com/rankonelabs/jig)
  agent produce EERs as part of its evaluation flow
- `eerful` CLI — `verify`, `publish-evaluator`
- `services/zg-bridge/` — TypeScript HTTP service that wraps the 0G
  broker + storage SDKs (the Python adapters drive it locally over
  loopback). See [`services/zg-bridge/README.md`](services/zg-bridge/README.md).

## Install

```bash
uv sync
```

Requires Python 3.12 and a 0G Galileo testnet wallet with faucet funds for
real TeeML calls and Storage uploads.

For producing receipts AND for the default verify path (which fetches
artifacts from 0G Storage), also start the bridge — see
[`services/zg-bridge/README.md`](services/zg-bridge/README.md) for
`npm install` + env vars. Fully-offline verification with `--bundle`,
`--report`, and `--skip-step-5` flags can avoid the bridge entirely.

The bridge listens on `127.0.0.1:7878` by default; the `eerful` CLI
refuses non-loopback bridge URLs unless you pass `--allow-remote-bridge`.
Library callers using `BridgeStorageClient` / `ComputeClient` directly
accept any URL — the loopback guard is CLI-only.

## Quick verify

```bash
eerful verify path/to/receipt.json
```

The verifier fetches the evaluator bundle and attestation report from
0G Storage by content hash (via the local `services/zg-bridge/`), runs
the §7.1 verification algorithm (Steps 1–6, with Step 5 in compose-hash
gating mode per §6.5), and prints a verdict plus the structured score
block. Step 7 (provider crosscheck) is deferred; full Step 5 (TDX/NVIDIA
chain to vendor roots) requires the dstack-verifier integration that's
out of scope for v0.4.

**Known limitation (cross-instance fetch):** the bridge can only fetch
content_hashes its own `uploadIndex` has seen — it can't recover the
0G rootHash from a sha256 alone. So a verifier on a fresh bridge
instance can't fetch a bundle that was uploaded by a different bridge
process. Workarounds today: (a) producer + verifier share a bridge,
(b) re-upload the bytes through the verifier's bridge to repopulate
the index, or (c) use `--bundle` / `--report` with local files. A
fix is planned (see `services/zg-bridge/README.md` Limitations).

`--bundle <path>` and `--report <path>` override individual artifacts
with local files (offline verification, cached blobs, or working around
the cross-instance limitation above); `--skip-step-5` skips the report
fetch entirely.

## What an EER proves (and doesn't)

A valid EER proves a specific response was produced by attested TEE hardware
under publicly defined evaluation criteria, with the response signed by an
enclave-born key. Authenticity is what EER provides.

It does **not** prove the evaluation is correct, the score is meaningful, the
producer chose representative input, or — most importantly for honest framing
— that the model the TEE loaded matches the model the evaluator declared.

That last gap is the subject of [spec §8](docs/spec.md#8-compose-vs-model-identity-binding),
which lays out the empirical state of 0G TeeML providers as a three-tier
spectrum:

- **Category A — bound launch string.** RTMR3 binds a compose whose launch
  command names the model identifier. The strongest model claim available
  today on widely deployed primitives (1 of 7 mainnet providers as of
  2026-04). Weights themselves still come from HuggingFace at runtime and
  are not measured.
- **Category B — unrelated compose.** RTMR3 binds a compose that does not
  reference the advertised model at all (3 of 7 — observed running a Phala
  demo Next.js starter). Receipts under such providers verify only that
  "some dstack TD ran some app."
- **Category C — centralized passthrough.** The compose attests a broker
  proxy that routes to a centralized backend with no TEE attestation (3 of
  7). The provider's own broker code admits this.

EER's protocol-level mitigation is the **`accepted_compose_hashes`** allowlist
(spec §6.5): an evaluator publisher pins the compose-hashes of providers whose
configuration they have inspected; verification fails closed when the attested
hash isn't in the list. `eerful verify` surfaces the §8 category and gating
status (`enforced` / `skipped`) on stdout by default — the report is fetched
from storage automatically.

Read §2 and §8 before relying on receipts. The spec is deliberate about what
an EER does and does not cryptographically establish.

## Producing receipts from a jig pipeline

`eerful.jig` plugs into [jig](https://github.com/rankonelabs/jig)'s
`LLMClient` / `Grader` / `TracingLogger` interfaces so any pipeline step
graded by an `EvaluationGrader` produces a verifying receipt as a side
effect.

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
            provider_address="0xd9966e13a6026Fcca4b13E7ff95c94DE268C471C",
        )
        grader = EvaluationGrader(client=client)

        config = PipelineConfig(
            name="trading-critic",
            steps=[Step(name="critic", fn=my_step, grader=grader)],
            tracer=SQLiteTracer("traces.db"),
        )
        result = await run_pipeline(config, input="market-neutral-v1")


asyncio.run(main())

# Each grader call produced an EER. The receipt is on the LLM_CALL span's
# metadata under `eerful.receipt_id`, and persisted to FeedbackLoop (when
# one is configured on the grader). Subsequent calls auto-chain via
# `previous_receipt_id`; pin per-call with provider_params.
```

The grader only emits scores for top-level numeric fields of the bundle's
`output_schema`; `score_dimensions=[...]` filters further. Tools, mismatched
system prompts, and conflicting inference params raise `EvaluationClientError`
— the receipt has to attest the bundle's criteria, not the caller's.

## Demo provider

The reference demo runs against **Provider 1 — `zai-org/GLM-5-FP8` at
`0xd9966e13a6026Fcca4b13E7ff95c94DE268C471C`**, the only acknowledged 0G TeeML
provider observed in Category A as of 2026-04. Its attested compose runs
vLLM with `--model zai-org/GLM-5-FP8` on the launch command, so RTMR3 binds
the model identifier as a string. The demo evaluator bundle pins this
provider's compose-hash in `accepted_compose_hashes` so receipts produced
against it pass Step 5 with `gating: enforced`.

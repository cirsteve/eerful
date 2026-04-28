# Trading-critic demo

The headline EER demo. Three iterations of a hand-authored trading
strategy, critiqued in sequence by a TEE-attested LLM, producing a
chain of three verifying receipts.

```
strategies/v1.md ─┐
                  ├─→ critic (TeeML, GLM-5-FP8) ─→ receipt v1.json ─┐
                  │                                                 │
strategies/v2.md ─┤                                                 ├─ chained by
                  ├─→ critic (TeeML, GLM-5-FP8) ─→ receipt v2.json ─┤  previous_receipt_id
                  │                                                 │
strategies/v3.md ─┤                                                 │
                  └─→ critic (TeeML, GLM-5-FP8) ─→ receipt v3.json ─┘
                                                       │
                                                       └── all three share one
                                                           input_commitment
                                                           (strategy_identity = "trading-strategy-v0")
```

This exercises every spec step in the v0.5 reference implementation
(Steps 1-6) plus the chain pattern (§6.6) and input-commitment
construction (§6.7).

## Setup

Bridge prerequisite — must be running before `demo.py` will get past
its first `healthz()` call:

```bash
cd services/zg-bridge && npm install && npm run dev
```

`.env` (at the repo root) needs `EERFUL_0G_PRIVATE_KEY` and
`EERFUL_0G_RPC` set. See `services/zg-bridge/README.md` for the
full bridge config surface. Wallet needs ≥ 1.05 0G on testnet
(broker MIN_LOCKED_BALANCE = 1.0 + a small buffer for inference).

Strategies v2 and v3 are gitignored — pre-generate them once before
running the demo:

```bash
export ANTHROPIC_API_KEY=...
uv run python examples/trading_critic/author_strategies.py --to v2
uv run python examples/trading_critic/author_strategies.py --to v3
# Hand-edit the generated files until they read cleanly.
```

`accepted_compose_hashes` in `bundle.json` ships as a syntactic
placeholder. Confirm Provider 1's live compose-hash and pin it
before publishing:

```bash
uv run python examples/trading_critic/bundle_inspect.py --confirm-compose-hash
# Paste the printed value into bundle.json.

uv run eerful publish-evaluator --bundle examples/trading_critic/bundle.json
```

## Run

```bash
uv run python examples/trading_critic/demo.py
```

Produces `receipts/{v1,v2,v3}.json` (gitignored). Idempotent: a
re-run with all three receipts already present and consistently
linked exits cleanly. A partial state forces a full restart.

`--verbose` dumps the critic's commentary alongside each score block.

## Verify

```bash
uv run python examples/trading_critic/verify_chain.py
```

Runs §7.1 Steps 1-6 against all three receipts and adds the
chain-specific assertions: forward `previous_receipt_id` linkage,
input-commitment stability, bundle stability, §8.2 Category A on
each report.

For single-receipt verification:

```bash
uv run eerful verify examples/trading_critic/receipts/v1.json
```

## What this proves CRYPTOGRAPHICALLY

- **Three real TeeML inference calls happened.** Steps 1, 4, 6 verify:
  receipt integrity, attestation report content hash, enclave
  signature over `response_content`. The provider's enclave-born key
  signed each response.
- **The provider sits in §8.2 Category A.** RTMR3 binds the attested
  `compose-hash`, and the compose declares the model identifier in
  its launch command. Step 5 enforces the gate — the bundle's
  `accepted_compose_hashes` allowlist has the live compose-hash
  pinned, and verification fails closed if Provider 1 rotates to a
  compose not in the list.
- **The chain is structurally consistent.** Each receipt's
  `previous_receipt_id` resolves; v1 is the root. A reviewer can re-run
  `verify_chain.py` and confirm.
- **The input commitment is stable across all three.** All three
  receipts carry the same non-None `input_commitment`, derived per
  spec §6.7 from `sha256(canonical_json({"strategy_identity": ...}) ||
  evaluator_id || salt)`. The producer can reveal the salt + identity
  to anyone who needs to verify the commitment matches.

## What this does NOT prove

- **The critic's scores are accurate.** EER attests the inference
  ran; not that the judgment was good. A poorly designed critic
  produces meaningless receipts. Higher-layer evaluator-reputation
  protocols are the right place for "is this critic trustworthy?"
- **The model the TEE loaded matched its expected weight bytes.** The
  attested compose-hash binds the launch *string* (the
  `--model zai-org/GLM-5-FP8` argument). Weights are HuggingFace-pulled
  at runtime and not measured; image tags may be unpinned. See spec
  §8.3 for what mitigations exist beyond the compose-hash allowlist.
- **The chain is complete.** The producer could have run ten
  critiques and chosen to publish three. EER per-receipt authenticity
  is the protocol's scope; chain completeness is a higher-layer
  concern (append-only logs, public registries, etc.).
- **The strategies are real or backtested.** They are hand-authored
  markdown, designed to give the critic differentiable material across
  iterations. No code execution, no live trading, no historical
  performance.
- **The input commitment binds the strategy *source*.** Per §6.7's
  producer-chosen rule, it binds the strategy *identity*. We
  documented our choice — `bundle.json` →
  `metadata.input_commitment_construction` — and the verifier can
  reconstruct it given the salt.

## Layout

| file | committed? | what |
|---|---|---|
| `bundle.json` | yes | EvaluatorBundle JSON. `accepted_compose_hashes` is the gate. |
| `strategies/v1.md` | yes | Hand-authored MA-crossover. |
| `strategies/v2.md` | no (gitignored) | Pre-generated by `author_strategies.py --to v2`. |
| `strategies/v3.md` | no (gitignored) | Pre-generated by `author_strategies.py --to v3`. |
| `author_strategies.py` | yes | Maintainer-only Anthropic helper for drafting v2/v3. |
| `bundle_inspect.py` | yes | Maintainer-only. Confirms Provider 1's compose-hash before publishing. |
| `demo.py` | yes | The chain orchestrator. |
| `verify_chain.py` | yes | Chain + commitment assertions (distinct from `eerful verify`). |
| `.salt` | no (gitignored) | Generated by `demo.py` first run; persists the chain's stable salt. |
| `receipts/{v1,v2,v3}.json` | no (gitignored) | Produced by `demo.py`. |

## Spec references

- §6.5 — evaluator bundle, `accepted_compose_hashes`
- §6.6 — receipt chains (`previous_receipt_id`)
- §6.7 — input commitment construction
- §7.1 — verification algorithm
- §8 — compose vs. model identity binding (most important context for
  reading the "What this proves / does NOT prove" lists honestly)

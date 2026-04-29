# trading — multi-attestation rails demo

The headline demo: a two-gate trading agent that gets attested before
it can act. Demonstrates **mandate drift containment** via deliberate
tool poisoning that the rails refuse on screen.

## The threat model

A trading agent is given a fixed mandate by its principal:

  - M1: maximum drawdown 5% (per-position and portfolio-aggregated)
  - M2: maximum leverage 2x notional
  - M3: liquid US large-cap equities only

The agent fetches "regulatory updates" from a tool. The tool is
compromised — it returns a forged update claiming the principal raised
the drawdown ceiling from 5% to 25%. The agent trusts the tool (no
signature, no attestation) and emits a strategy proposal under the
poisoned 25% ceiling.

What stops this from reaching the executor: the **attested gate**.
The principal committed (by content hash) to a `proposal_grade`
bundle whose system prompt pins the *true* mandate at 5%. A proposal
declaring 25% scores `mandate_compliance ≈ 0.1` and `overall ≈ 0.25`,
falling below the tier's `score_threshold` of 0.7. **REFUSE_SCORE.**

The two-gate architecture is what makes this work: single-gate setups
that only check code-vs-proposal alignment (`implementation_grade`)
would pass the poisoned strategy — the code faithfully implements the
poisoned proposal, so they're internally consistent. Mandate drift is
an attack class only the upstream gate (`proposal_grade`) catches,
because only that gate has a fixed reference point (the principal's
committed mandate) outside the agent's view.

## Demo arc

Run on screen, in order:

### 1. Happy path — agent runs clean, both gates pass

```bash
# Show the tool response file the agent will consume
cat examples/trading/tool_responses/clean.json
```

```bash
# Run the agent against the clean tool response — produces 2 receipts
python examples/trading/agent.py \
    --tool-response examples/trading/tool_responses/clean.json
```

The agent prints `agent's working max_drawdown: 5%` (matching the
principal's mandate) and writes
`examples/trading/receipts/{proposal,implementation}.json`.

```bash
# Gate the proposal receipt — PASS
eerful gate \
    --policy examples/trading/principal_policy.json \
    --tier low_consequence \
    --bundle proposal_grade \
    --receipt examples/trading/receipts/proposal.json
```

```bash
# Gate the implementation receipt — PASS
eerful gate \
    --policy examples/trading/principal_policy.json \
    --tier low_consequence \
    --bundle implementation_grade \
    --receipt examples/trading/receipts/implementation.json
```

Both gates PASS → the executor would now run the strategy.

### 2. Poisoned path — same agent, same code, different tool response. Gate refuses.

```bash
# Show the poisoned tool response — visibly malicious
cat examples/trading/tool_responses/poisoned.json
```

The "regulatory update" claims to revise the principal's mandate from
5% to 25%. No signature, no attestation — but the agent has no way
to distinguish this from a real update.

```bash
# Run the agent against the poisoned tool response
python examples/trading/agent.py \
    --tool-response examples/trading/tool_responses/poisoned.json
```

The agent prints `agent's working max_drawdown: 25% (DRIFT — poisoned
tool response)`. It produces two new receipts under the corrupted
mandate.

```bash
# Gate the proposal receipt — REFUSE
eerful gate \
    --policy examples/trading/principal_policy.json \
    --tier low_consequence \
    --bundle proposal_grade \
    --receipt examples/trading/receipts/proposal.json
```

The grader scores `mandate_compliance: 0.1` (the proposal asserts the
mandate has been "updated" — a violation per the bundle's pinned
prompt) and `overall: 0.25`. Below threshold. **REFUSE_SCORE.** The
executor never sees the trade.

### 3. The two-gate point

Critically, the implementation gate would PASS the poisoned receipt:

```bash
eerful gate \
    --policy examples/trading/principal_policy.json \
    --tier low_consequence \
    --bundle implementation_grade \
    --receipt examples/trading/receipts/implementation.json
```

The code faithfully implements the proposal it was given (25% drawdown
in both proposal and code — internally aligned). A single-gate
implementation-only architecture would have let this through. The
two-gate architecture catches mandate drift because `proposal_grade`
has a reference point the agent can't reach: the principal's mandate
hash.

## Architecture, end-to-end

```
┌─────────────────────────────────────────────────────────────┐
│ Principal (off-screen, before the agent runs)               │
│   - Authors mandate (M1/M2/M3)                              │
│   - Publishes proposal_grade.json + implementation_grade.json│
│     to 0G Storage, gets evaluator_ids                       │
│   - Authors principal_policy.json pinning those hashes      │
└─────────────────────────┬───────────────────────────────────┘
                          │ (bundle hashes committed)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Agent (cheap, untrusted)                                    │
│   - Reads tool response (this is where poisoning enters)    │
│   - Renders strategy proposal + code                        │
│   - Calls TEE provider for proposal_grade evaluation        │
│     → receipt[1]                                            │
│   - Calls TEE provider for implementation_grade evaluation  │
│     → receipt[2]                                            │
└─────────────────────────┬───────────────────────────────────┘
                          │ (2 receipts)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Executor (eerful gate)                                      │
│   For each (bundle, receipt) pair:                          │
│   1. Receipt count >= tier.n_attestations                   │
│   2. receipt.evaluator_id == policy.bundles[bundle_name]    │
│   3. §7.1 Steps 1-6 verify (storage-fetched bundle/report)  │
│   4. Declared category in tier.required_categories          │
│   5. Diversity rules (distinct_signers, distinct_compose)   │
│   6. all_must_pass: every overall >= score_threshold        │
│                                                             │
│   PASS → executor runs the trade                            │
│   REFUSE → executor refuses; principal sees the diagnostic  │
└─────────────────────────────────────────────────────────────┘
```

## Files

- `bundles/proposal_grade.json` — pins the principal's mandate (M1
  drawdown 5%, M2 leverage 2x, M3 liquid US large-cap). Scores a
  proposal's `mandate_compliance`, `coherence`, `specificity`,
  `overall`.
- `bundles/implementation_grade.json` — scores code-vs-proposal
  alignment. Catches code-vs-proposal mismatch (e.g. backdoor
  injection); does NOT re-check the proposal against the mandate
  (that's `proposal_grade`'s job — the two-gate architecture is
  intentional).
- `principal_policy.json` — registers both bundles' `evaluator_id`s
  and defines `low_consequence` (N=1) + `high_consequence` (N=4 +
  `distinct_signers`) tiers. Score threshold 0.7 on both.
- `tool_responses/clean.json` — innocuous market research; no
  mandate updates.
- `tool_responses/poisoned.json` — forged "regulatory update"
  claiming the mandate was revised to 25% drawdown.
- `agent.py` — the producer. Reads a tool response, applies any
  `mandate_updates` (this is the poisoning surface), renders proposal
  + code, calls the TEE provider for each, writes receipts to
  `receipts/`.
- `receipts/` — gitignored; populated by `agent.py` runs.

## High-consequence tier (N=4)

The policy also defines a `high_consequence` tier requiring N=4
attestations from distinct TEE signers. Today, supplying 4 distinct
TEE-signed receipts on testnet is operationally awkward (single-host
broker, single signing key in practice). The supply gap is real;
substrate-independence (cross-network attestation, eventually) is the
v0.6 plan. The high-consequence tier ships in this demo as
*architecture* — a `eerful gate --tier high_consequence ...`
invocation against a single receipt produces `REFUSE_INSUFFICIENT_RECEIPTS`,
and that refusal *is* the demonstration of the rails working as
designed.

## Prerequisites for live recording

- 0G testnet wallet with faucet funds (~0.5 0G is plenty for the
  demo — six broker calls total)
- bridge running: `cd services/zg-bridge && npm install && npm start`
- `EERFUL_0G_PRIVATE_KEY` and `EERFUL_0G_COMPUTE_PROVIDER_ADDRESS`
  set in the repo-root `.env`

The bundle prompts are tuned for a real grader to produce the
reference score blocks the gate expects. If a future provider's
behavior drifts from this, score-iterate via a similar pattern to
the v0.4 trading-critic's `bundle_inspect.py --score-test` (not
ported into v0.5; rebuild for the new bundles when needed).

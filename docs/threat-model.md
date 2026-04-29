# eerful — threat model

The defense the eerful rails provide and the attacks they don't. A
single-page reference; the spec ([`docs/spec.md`](spec.md)) covers the
receipt-level threat model in §9.2, this doc lays out the rails-level
context.

## Premise

Autonomous agents in 2026 will get compromised. The compromise vectors
are not hypothetical:

- **Prompt injection through retrieved content.** Documents the agent
  retrieves contain instructions the agent treats as authoritative.
- **Tool poisoning.** A malicious or compromised tool returns a forged
  response (e.g. a "regulatory update" that revises the agent's
  working mandate).
- **Adversarial RAG.** A corpus the agent searches contains content
  authored to manipulate the agent's reasoning.
- **Supply-chain compromise of cheap model weights.** Local or
  open-source models the agent uses for cheap exploration are not
  audited at the bit level.
- **Composability bugs.** Agents calling agents create paths where the
  inner agent's compromise leaks into the outer agent's outputs.

The defensive posture eerful assumes is: **don't try to make the
exploration layer secure**. Make it cheap and fast and creative,
expect it to be compromisable, and place an attested gate before any
compromise can have consequences.

## What the rails defend against

**1. Mandate drift.** A poisoned tool response causes the agent to
violate criteria the principal committed to. eerful's
`proposal_grade`-class bundles pin the principal's criteria in the
bundle's `system_prompt`. The bundle hash is content-addressed
(`evaluator_id`) and committed in the principal's policy
*before the agent runs*. A compromised agent that proposes a strategy
violating the pinned criteria scores low on `mandate_compliance`; the
executor refuses below threshold. The agent has no path to the
bundle's reference point, only the executor does.
*Demonstrated by [`examples/trading/`](../examples/trading/) — the
poisoned-tool path that the rails refuse on screen.*

**2. Code-vs-spec backdoor injection.** A poisoned tool response or
adversarial RAG content causes the agent to add hidden logic outside
the proposal's stated scope (a hardcoded recipient address, a
conditional override of risk controls). eerful's
`implementation_grade`-class bundles score code-vs-proposal
alignment; logic outside the proposal's scope drops `spec_alignment`
and the executor refuses. *Two-gate architecture is essential here:
mandate drift is caught upstream by `proposal_grade`, backdoor
injection is caught by `implementation_grade`. Either bundle alone
misses one of the two attack classes.*

**3. Compromised single TEE in a high-consequence call.** A
high-consequence policy demands N receipts from N distinct
`tee_signer_address`es (per `diversity.distinct_signers`). A single
compromised TEE can produce receipts that pass Steps 1-6 of EER
verification but cannot satisfy the diversity rule on its own. The
attacker has to compromise N independent TEEs.

**4. Substrate substitution.** A receipt produced under a different
evaluator than the principal committed to fails Check 2 of the gate
(`receipt.evaluator_id != policy.bundles[bundle_name]`). The principal's
pre-commitment to bundle hashes is the load-bearing trust anchor.

**5. Score forgery.** A receipt with a fabricated score block fails
Step 1 (`receipt_id` recomputation) or Step 6 (signature recovery
against `enclave_pubkey`). EER receipts are tamper-evident at the
field level.

## What the rails do NOT defend against

**1. The model loaded a different weight bundle than the
`model_identifier` declares.** This is the §8 gap and it's
substrate-bounded: a TEE compose can name a model on the launch
command line (Category A), but weights typically come from
HuggingFace at runtime and are not measured into RTMR3. The
`accepted_compose_hashes` allowlist + tier-level
`required_categories` enforcement reduce this to "the publisher
trusts this provider's compose configuration"; closing the gap
fully requires upstream changes to the inference container (in-enclave
weight measurement; spec §8.4).

**2. The principal's *criteria* are wrong.** EER attests that
evaluation happened against the bundle's published criteria. If the
criteria themselves are flawed, receipts are authentic but the
evaluations are meaningless. This is a higher-layer concern: criteria
are public and content-addressed, anyone can read them before relying
on receipts.

**3. The agent omits unfavorable evaluations.** A producer who runs ten
gate calls and submits the three favorable receipts has produced
authentic receipts for those three. Chain completeness is
producer-asserted (spec §6.6). Mitigation lives in append-only
publication or external sequencing — not in the receipt format.

**4. Replay attacks at the agent layer.** A receipt is a statement
about an evaluation, not about a producer or a deployment. Higher
layers that bind receipts to producers (NFT ownership, marketplace
registration, on-chain anchoring) provide replay resistance through
the binding mechanism. EER does not.

**5. The principal is the attacker.** A principal who authors a
permissive policy, low score thresholds, and broad
`accepted_compose_hashes` will pass any receipt. The rails enforce
what the principal commits to; they do not adjudicate whether the
commitment was wise.

**6. Side-channel leakage from the TEE.** The TDX trust boundary
includes the platform vendor (Intel for TDX, NVIDIA for H100 GPU
attestation). Compromise of the platform's signing root or of the
hardware's cryptographic isolation breaks the receipt-authenticity
chain. The same applies to the inference framework (vLLM, sglang)
running inside the TD; a malicious framework version with the right
compose-hash is indistinguishable from a benign one at the EER layer.

## Defense composition

- **Per-receipt authenticity** — EER §7.1 verification (Steps 1-6)
- **Compose binding** — `accepted_compose_hashes` (§6.5) gates Step 5
  on publisher-reviewed compose hashes
- **Category enforcement** — executor's `required_categories` filters
  on publisher-declared §8.2 category
- **Provider diversity** — executor's `distinct_signers` /
  `distinct_compose_hashes` for high-consequence tiers
- **Two-gate architecture** — separate `proposal_grade` and
  `implementation_grade` bundles catch mandate drift and backdoor
  injection respectively; either alone is insufficient
- **Score aggregation** — `all_must_pass` requires every receipt's
  `overall` to clear the tier threshold; one TEE saying no blocks the
  set

Each layer composes with the others. The rails don't pretend any
single layer is sufficient — `examples/trading/`'s mandate-drift demo
is exactly the case that two-gate + score threshold + bundle-hash
commitment together catch, and that any one of them alone misses.

## Substrate

This implementation runs against 0G's TeeML / Storage primitives on
testnet. The architecture is substrate-independent: the executor
reads receipts and policies, not substrate. If 0G's ecosystem matures
(per spec §8.2, the Cat A supply gap is real today), the rails
enforce against it. If the right substrate turns out to be Phala,
Marlin, AWS Nitro, or something not yet shipped, the executor ports
without changes.

## Where this doc lives

This is a one-page reference; if you want the full receipt-format
threat model see [`docs/spec.md`](spec.md) §9.2, and if you want the
threat model dramatized in working code see
[`examples/trading/`](../examples/trading/) — both paths are
recordable end-to-end.

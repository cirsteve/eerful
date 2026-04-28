# Enhanced Evaluation Receipts (EER)
## Draft Specification (v0.5)

---

## 1. Overview

Enhanced Evaluation Receipts (EER) define a portable, verifiable artifact for recording the result of an LLM evaluation executed inside a Trusted Execution Environment (TEE).

An EER binds together four elements:

- A **public evaluator definition**, content-addressed and retrievable, that anyone can inspect.
- A **private input commitment**, optionally produced by the producer, that links a receipt to an input identity without disclosing the input.
- An **attested model execution**, performed by a TEE-hosted inference provider, with a hardware-rooted signature over the response.
- An optional **structured evaluation output**, parsed from the response according to a schema declared by the evaluator.

The result is a self-contained receipt that can be verified by a third party without requiring access to the original input, and without depending on the inference provider remaining online at verification time.

EER sits between two neighboring techniques: it is **stronger than logs** (signed instead of asserted, content-addressed instead of host-replicated) and **weaker than ZK proofs** (trusts hardware vendors rather than reducing to mathematics), and it is **practical today** on widely deployed primitives. Where ZK is unavailable or impractically expensive — most LLM workloads — EER is the strongest available evaluation credential.

EER is built on the 0G Compute Network's TeeML inference primitive and 0G Storage. Other TEE-attested inference and content-addressed storage protocols can be substituted; the EER schema and verification algorithm are protocol-agnostic at the receipt level.

---

## 2. What an EER Proves

A valid EER proves:

- A specific response was produced by attested TEE hardware.
- The response was signed by a key bound to that hardware via the vendor's attestation chain.
- The producer claims this response was generated under publicly defined evaluation criteria, referenced by content hash and retrievable from storage.

A valid EER does NOT prove:

- The evaluation is correct or the score is meaningful.
- The model the TEE loaded matches the model the evaluator declares (see §8.3).
- The producer used honest, well-formed, or representative input.
- The receipt represents a complete history (see §6.6 on chain semantics).
- The producer did not run additional evaluations and discard unfavorable receipts.

Verifiers reading a receipt should hold these distinctions actively. Authenticity is what EER provides; correctness, completeness, and good-faith use are higher-layer concerns.

---

## 3. Design Goals

EER is designed to provide:

1. **Execution Authenticity.** A verifier can confirm that the response was produced by genuine TEE hardware running on a known provider, signed by an enclave-born key.
2. **Evaluator Transparency.** The criteria under which a producer was evaluated are publicly committed by content hash. Anyone holding a receipt can fetch the evaluator and read what standard was applied.
3. **Input Privacy.** The producer's input passes through the TEE encrypted in transit and is processed in hardware-isolated memory. The receipt does not disclose the input. An optional input commitment links receipts to the same input without revealing it.
4. **Portable Verification.** All verification material is either inside the receipt or retrievable from content-addressed storage by hashes the receipt names. Verification is local; no consensus, no validator network, no provider liveness required.
5. **Composability.** EERs chain via `previous_receipt_id`, allowing producers to publish a verifiable evolution of an evaluated artifact over time. Receipts compose with downstream agents, reputation systems, and credential consumers without modification.

---

## 4. Non-Goals

EER does NOT guarantee:

- **Correctness of the evaluation.** A flawed evaluator produces flawed scores. Receipts attest that evaluation happened; not that it was right.
- **Quality or truthfulness of model outputs.** TEE attestation binds output to enclave, not output to truth.
- **Deterministic reproducibility.** LLM inference is non-deterministic; two evaluations of the same input under the same evaluator may produce different receipts. EER attests authenticity per call, not agreement across calls.
- **Trustlessness.** Verification requires trusting the TEE hardware vendors (Intel TDX, NVIDIA H100/H200) and the attestation chains they publish. Receipts are no stronger than the hardware roots they depend on.
- **Validator-network consensus.** EER is a credential protocol, not a consensus protocol. A receipt represents a single producer's claim about a single execution. Aggregating receipts into reputation, dispute resolution, or marketplace ranking is out of scope and left to higher layers.

---

## 5. System Model

EER defines four roles. A given party MAY play multiple roles.

### 5.1 Producer

The party whose private input is being evaluated.

- **Holds:** the private input (strategy, code, model weights, prompt, etc.); a wallet for paying the compute provider; optionally the input's prior receipts.
- **Produces:** the EER artifact, by orchestrating an evaluator fetch, an attested inference call, and the assembly of the receipt fields.
- **Trusts:** the TEE hardware vendors' attestation chains; the evaluator publisher's good faith in declaring criteria honestly.
- **Does not trust:** the compute provider with the contents of the input (the TEE protects against this); the evaluator publisher to retroactively change criteria (the content hash protects against this).

### 5.2 Evaluator Publisher

The party that authors and publishes a public criteria bundle.

- **Produces:** an evaluator bundle (see §6.5) containing the system prompt, model identifier, output schema, and version metadata. Uploads it to content-addressed storage. Distributes the resulting `evaluator_id` (the content hash) so producers and verifiers can reference it.
- **Trusts:** the storage layer to keep the bundle retrievable.
- **Does not trust:** producers to apply the criteria fairly to their own inputs (the evaluator bundle is public and content-addressed, so producers cannot modify it).

The evaluator publisher's reputation is independent of EER. A flawed or malicious evaluator publishes flawed criteria; receipts produced under it are authentic but useless. Higher-layer protocols may track evaluator reputation; EER does not.

### 5.3 Compute Provider

A 0G Compute Network inference provider operating a TeeML service.

- **Holds:** TEE hardware (Intel TDX CPU + NVIDIA H100/H200 GPU); an enclave-born signing keypair; published attestation reports binding the pubkey to the hardware.
- **Produces:** signed inference responses; downloadable signature artifacts keyed by `chat_id`; downloadable attestation reports.
- **Trusts:** standard 0G broker mechanics for billing and provider acknowledgment.
- **Does not trust:** the producer to be honest about input contents; the producer to pay (handled by the broker's escrow).

The compute provider's role is fixed by the 0G Compute Network protocol. EER does not extend it.

### 5.4 Verifier

A third party presented with a receipt who wishes to verify it.

- **Holds:** the receipt; access to content-addressed storage to fetch evaluator bundles and attestation reports; the public root certificates of the TEE vendors (Intel, NVIDIA).
- **Produces:** a binary verdict — receipt is valid, or it is not — plus, on success, the structured score and verified evaluator metadata.
- **Trusts:** the same hardware roots as the producer.
- **Does not trust:** the producer's claims about the input or the result; verification is meant to confirm what the receipt actually establishes, independent of producer assertions.

A verifier may operate offline relative to the compute provider — all verification material is in the receipt or in storage at hashes the receipt names.

---

## 6. Receipt Structure

### 6.1 Schema

```
EnhancedReceipt {
  # Identity (derived; not in canonical signing payload)
  receipt_id                : Bytes32Hex   # sha256 of canonical_signing_payload(receipt)

  # Spec version (in canonical signing payload, see §10.2)
  spec_version              : str          # "0.5"; verifiers refuse mismatching versions

  # Producer claims (in canonical signing payload)
  created_at                : datetime     # UTC, validated tz-aware UTC at construction
  evaluator_id              : Bytes32Hex   # content hash of evaluator bundle on Storage
  evaluator_storage_root    : Bytes32Hex   # backend retrieval locator for the bundle bytes
  evaluator_version         : str          # human-readable version, denormalized from bundle
  input_commitment          : Bytes32Hex?  # producer-chosen hash of the private input; nullable
  previous_receipt_id       : Bytes32Hex?  # links to predecessor receipt; nullable

  # Compute provider attribution (in canonical signing payload)
  provider_address          : Address      # 0G Compute provider's wallet address
  chat_id                   : str          # 0G's identifier for the inference call
  response_content          : str          # the LLM output text

  # Structured evaluation (in canonical signing payload)
  output_score_block        : dict?        # parsed from response_content per evaluator schema; nullable

  # Attestation report identity (in canonical signing payload, see §6.3)
  attestation_report_hash   : Bytes32Hex   # content hash of provider's attestation report on Storage
  attestation_storage_root  : Bytes32Hex   # backend retrieval locator for the report bytes

  # Attestation block (NOT in canonical signing payload — the signature cannot be over itself)
  enclave_pubkey            : BytesHex     # provider's enclave-born public key, cached (see §6.4)
  enclave_signature         : BytesHex     # signature over response_content, cached for offline verify (see §6.4)

  # Extensions (in canonical signing payload, see §10.1)
  extensions                : dict?        # namespaced extension fields; nullable
}
```

`Bytes32Hex` is a 0x-prefixed 64-character lowercase hex string. `BytesHex` is a 0x-prefixed lowercase hex string of variable length, used for fields whose byte width depends on the underlying cryptographic primitive (e.g. `enclave_pubkey`, `enclave_signature`). `Address` is a 0x-prefixed 40-character hex string (EVM address). Optional fields use `?`. All hex-typed fields are encoded as lowercase per §6.4; the in-memory representation is implementation-defined, but the canonical wire and signing form is hex. Implementations MUST canonicalize to lowercase hex before any byte-comparison or hashing operation.

**Storage-root pairs.** Each storage-resident artifact (the evaluator bundle, the attestation report) is referenced by a *pair* of fields:

- An **integrity hash** (`evaluator_id`, `attestation_report_hash`) — sha256 of the canonical bytes. The verifier MUST re-hash the fetched bytes and confirm equality.
- A **retrieval locator** (`evaluator_storage_root`, `attestation_storage_root`) — the backend's lookup key. For 0G Storage this is the Merkle rootHash; for an alternate backend it is whatever the backend's content-addressing scheme produces. Both fields are required (§7.1 Steps 2 and 4).

The pair lets any verifier with any storage instance fetch any receipt's bytes without depending on producer-side state (e.g., a sha256→rootHash index that resets on bridge restart). The integrity hash is what binds the receipt to specific bytes; the retrieval locator is what makes the bytes findable. They are not redundant.

`Bytes32Hex` is the v0.5 type for both retrieval locators because the profiled 0G Storage backend uses 32-byte Merkle roots. This should not be read as a claim that all alternate backends' native lookup keys fit `Bytes32Hex`: standard IPFS CIDs and standard Arweave TX IDs do not, as encoded, have that form. A backend whose native locator is not a 32-byte hex value would require a widened locator type (for example `String` or `BytesHex`) together with an explicit canonical encoding rule; adopting such a locator would be a version-coordinated canonical-form change.

### 6.2 Field semantics

`receipt_id` is the canonical identity of the receipt, derived from a deterministic hash of the signed payload. Two receipts with identical signed-payload contents have identical `receipt_id`s. Verifiers MUST recompute `receipt_id` from the payload and reject the receipt if it does not match.

`created_at` is a UTC timestamp set by the producer at receipt construction, validated to be timezone-aware UTC. It is included in the signed payload. Backdating is structurally possible — the TEE attests the response, not the timestamp. Verifiers SHOULD compare `created_at` against external timing signals (block timestamps, log entries, the producer's first publication) when temporal claims matter.

`evaluator_id` is the content hash of the evaluator bundle (§6.5) the producer asserts was used. Verifiers MUST fetch the bundle from Storage by `evaluator_id`, recompute its hash, and confirm. The bundle declares the model identifier and output schema; both bind into verification.

`evaluator_version` is a human-readable version string, denormalized from the bundle for convenience. It is informational; verification trusts `evaluator_id`, not the version string.

`input_commitment` is OPTIONAL and producer-chosen. When present, it is a hash committing the producer to a specific private input identity, allowing receipts to be linked across time without disclosing the input. When absent, each receipt is unlinkable. See §6.7 for the recommended construction.

`provider_address` is the 0G Compute Network wallet address of the provider that ran the inference. Verifiers use this to look up the provider's published attestation report and to confirm `enclave_pubkey` is the one this provider has registered.

`chat_id` is the inference call's identifier as returned by 0G in the `ZG-Res-Key` response header (or `data.id` for chatbot responses). It is informational at verification time — the cached signature is what verifies, not a re-fetch — but `chat_id` allows a verifier with provider access to cross-check by re-fetching the signature from the provider.

`response_content` is the text the model produced. It is the body of what `enclave_signature` signs. Verifiers MUST verify the signature over the exact `response_content` bytes, using `enclave_pubkey`.

`output_score_block` is OPTIONAL structured data parsed from `response_content` per the evaluator bundle's `output_schema`. When present, it MUST validate against the bundle's schema; verifiers MUST fail the receipt if validation fails. When absent, the receipt is unstructured — the response text is the result, no machine-readable score is asserted.

`enclave_pubkey` and `enclave_signature` are cached at receipt construction. Caching them is what makes verification provider-independent — a verifier doesn't need the provider to be online. Verifiers MUST confirm `enclave_pubkey` matches the pubkey bound by the attestation report at `attestation_report_hash`.

`attestation_report_hash` is the content hash of the provider's CPU+GPU attestation report at the time the receipt was produced. The report is uploaded to Storage by the producer (or fetched and uploaded by tooling on the producer's behalf) so that verification doesn't require fetching from the provider. Reports MAY rotate; the hash pins which report this receipt's verification depends on.

`previous_receipt_id` is OPTIONAL and links this receipt to a predecessor — typically a receipt about the same input or the same evaluated artifact at an earlier time. See §6.6.

### 6.3 Canonical signing payload

The `enclave_signature` is over `response_content` only — this is what the 0G TeeML primitive natively signs.

The `receipt_id` is over a broader canonical payload that includes the producer's claims, the compute provider's attribution, the attestation report identity, the storage-root locators, the spec version, and any extension fields. The canonical signing payload for `receipt_id` derivation is the receipt with the following fields excluded:

- `receipt_id` itself (derived from this payload — chicken-and-egg).
- `enclave_pubkey` and `enclave_signature` (the signature over `response_content`; cached separately and verified by its own algorithm at Step 6).

What remains, encoded as canonical JSON (see §6.4): `attestation_report_hash`, `attestation_storage_root`, `chat_id`, `created_at`, `evaluator_id`, `evaluator_storage_root`, `evaluator_version`, `extensions` (when present; see §10.1), `input_commitment`, `output_score_block`, `previous_receipt_id`, `provider_address`, `response_content`, and `spec_version`.

`receipt_id = sha256(canonical_json(signing_payload))`.

Including `attestation_report_hash` in the signing payload binds the receipt to a specific attestation report, not just to a pubkey. Without this binding, a producer could substitute a different valid report binding the same pubkey post-construction; the substitution would not change `receipt_id` or invalidate `enclave_signature`, but it would change which Step-5 evidence verifiers fetch. Including the report hash forecloses this same-key/different-report swap.

Including the storage-root pair (`evaluator_storage_root`, `attestation_storage_root`) in the signing payload similarly forecloses post-construction substitution of a different rootHash that points at bytes which happen to hash to the same sha256. A producer who tried to swap in a separately-uploaded duplicate copy at a different rootHash would invalidate `receipt_id`. Note this binding is structural, not adversarial-trustworthy on its own: the producer can still freely choose any (matching) rootHash at construction time. What the binding buys is integrity *over time* — once the receipt is signed, the locator cannot drift without detection.

Including `spec_version` in the signing payload makes a version downgrade detectable: a verifier under v0.5 evaluating a payload that claims v0.4 fails Step 1 (the v0.4 form would not include the new fields, so the digest cannot match). See §10.2 for the asymmetric-break semantics.

### 6.4 Canonical encoding

For deterministic hashing, fields are encoded as canonical JSON:

- Object keys in lexicographic byte order.
- No insignificant whitespace.
- UTF-8 encoding.
- `null` for absent optional fields (not omission).
- Timestamps as RFC 3339 strings with explicit `Z` suffix for UTC.
- Bytes fields as 0x-prefixed lowercase hex.

Implementations MUST produce byte-identical output across runs and across implementations. This is normative.

EIP-712 typed-data encoding is a candidate for a future version to enable native EVM contract verification, additive over the canonical-JSON form. v0.5 ships canonical JSON for implementation simplicity.

### 6.5 Evaluator bundle

An evaluator bundle is a JSON document published to content-addressed storage. The `evaluator_id` is its content hash.

```
EvaluatorBundle {
  version                  : str            # human-readable, e.g. "trading-critic@1.2.0"
  model_identifier         : str            # e.g. "zai-org/GLM-5-FP8"
  system_prompt            : str            # the criteria, in natural language
  output_schema            : dict?          # JSON Schema for output_score_block; nullable
  inference_params         : dict?          # temperature, max_tokens, etc.; nullable
  accepted_compose_hashes  : [Bytes32Hex]?  # allowlist of attested compose-hashes; nullable
  metadata                 : dict?          # publisher-defined; informational
}
```

`model_identifier` MUST match an identifier the 0G Compute Network recognizes for verifiable services. The TeeML attestation does not bind the loaded model weights to this identifier — see §8 for what attestation does bind, and what it does not.

`output_schema`, when present, is a JSON Schema document that `output_score_block` MUST validate against. Verifiers MUST run schema validation as part of receipt verification when `output_score_block` is present.

`inference_params`, when present, fix the inference parameters (temperature, top-p, max_tokens, etc.) the producer SHOULD use. They are not enforced at the protocol level — the TEE doesn't attest to them — but consistent params across producers using the same evaluator improve cross-receipt comparability.

`accepted_compose_hashes`, when present, is an allowlist of TEE compose-hashes the publisher has reviewed and accepts as valid environments for producing receipts under this evaluator. Verifiers MUST extract the attested compose-hash from the attestation report (§7.1 Step 5) and reject the receipt if it is not in the allowlist. When absent, no compose-hash gating is performed and the receipt's model-environment claim rests on the protocol-level attestation alone (see §8). Publishers SHOULD populate this field when they have specific providers in mind whose compose configuration they have inspected; this is the strongest practical defense against the §8 gap available without upstream protocol changes.

Bundles are immutable after publication. A new version of an evaluator publishes a new bundle with a new `evaluator_id`. Receipts under the old `evaluator_id` remain valid under their evaluator; cohort comparison across `evaluator_id`s is a higher-layer concern.

### 6.6 Receipt chains

`previous_receipt_id` is OPTIONAL and links this receipt to a predecessor.

Typical use: a producer iterating on an evaluated artifact — strategy v1 receipt → strategy v2 receipt → strategy v3 receipt — chains receipts to publish a verifiable history.

Semantics:

- **Producer-asserted.** The TEE does not attest to chain structure; it only attests individual receipts. A producer who runs ten evaluations and publishes only the favorable three has produced authentic receipts, but the chain is incomplete. EER does not detect this.
- **Non-authoritative.** A presented chain MUST NOT be treated as a complete history without external guarantees (append-only public log, bilateral commitment, on-chain anchoring).
- **Forks permitted.** Two receipts MAY both reference the same `previous_receipt_id`. EER does not specify which is canonical; that is a higher-layer policy decision.
- **Cross-evaluator chains permitted.** A receipt under evaluator A MAY reference a previous receipt under evaluator B. The chain documents temporal ordering of evaluations of the same artifact, not directly comparable scoring.

A future version MAY add chain-completeness primitives (e.g., publication to an append-only log with on-chain root anchoring). v0.5 leaves this to higher layers.

### 6.7 Input commitment construction

When a producer chooses to populate `input_commitment`, the RECOMMENDED construction is:

```
input_commitment = sha256(input_bytes || evaluator_id || salt)
```

Where:

- `input_bytes` is a canonical byte serialization of the private input. The producer is responsible for choosing and documenting their canonicalization (e.g., serialized strategy code, a content hash of an opaque blob, a Merkle root over a structured artifact).
- `evaluator_id` is the receipt's `evaluator_id` field, included to bind the commitment to the evaluation context. Without this binding, a producer could reuse the same commitment across evaluators in ways that confuse linkage.
- `salt` is a producer-chosen random value. It prevents brute-force reversal of the commitment for low-entropy inputs and prevents trivial linkage across producers who happen to evaluate the same artifact.

Producers MAY use alternative constructions if their use case requires (e.g., commit-reveal schemes, Merkle proofs over input components). When deviating from the recommended construction, producers SHOULD document the construction in the receipt's evaluator bundle metadata or in a higher-layer registry, so verifiers know how to interpret the commitment.

The salt SHOULD be retained by the producer if they want to later reveal the input — without the salt, the commitment is a one-way hash and the input cannot be proven against it. Lost-salt commitments are recoverable as identity (linking receipts about the same input) but not as proofs (showing what the input actually was).

EER does not specify or check the construction. The commitment is producer-asserted; verifiers know the *value* but cannot independently verify it represents what the producer claims it represents. This is by design — producer privacy is the goal, and any protocol-enforced construction would constrain that.

---

## 7. Verification

A verifier presented with an `EnhancedReceipt` performs the following steps in order. The receipt is invalid if any step fails.

### 7.1 Verification algorithm

**Step 1: Receipt integrity.**
Compute `expected_receipt_id = sha256(canonical_json(signing_payload(receipt)))`. Confirm `expected_receipt_id == receipt.receipt_id`.
*Failure mode:* receipt was constructed with a non-canonical encoding, or has been tampered with, or `receipt_id` was set incorrectly.

**Step 2: Fetch and verify evaluator bundle.**
Fetch the evaluator bundle from content-addressed storage by `receipt.evaluator_storage_root` (the backend's retrieval locator). Compute the fetched bundle's sha256; confirm it matches `receipt.evaluator_id` (the integrity hash). Both fields are required: the locator is what the backend keys by, the hash is what binds the receipt to specific bytes.
*Failure mode:* bundle is unavailable at the named locator, or storage returned bytes that don't hash to `receipt.evaluator_id`.

**Step 3: Validate output schema (if applicable).**
If the evaluator bundle declares an `output_schema` AND the receipt has an `output_score_block`, validate the score block against the schema.
*Failure mode:* score block is malformed, missing required fields, or has the wrong types — the producer parsed the response incorrectly or the model produced output that doesn't fit the schema.

**Step 4: Fetch and verify attestation report.**
Fetch the attestation report from storage by `receipt.attestation_storage_root` (the backend's retrieval locator). Compute the fetched report's sha256; confirm it matches `receipt.attestation_report_hash` (the integrity hash). Symmetric to Step 2's bundle fetch.
*Failure mode:* report is unavailable at the named locator, or storage returned bytes that don't hash to `receipt.attestation_report_hash`.

**Step 5: Verify the attestation chain.**
Verify the report's CPU TDX quote against Intel's published root certificates. Verify the report's GPU attestation against NVIDIA's GPU Attestation API. Confirm the report binds a specific public key to the attested hardware. Confirm that bound key equals `receipt.enclave_pubkey`. Extract the report's `compose-hash` measurement (the value extended into RTMR3 by the dstack event log, equal to `sha256(app_compose)`); if the evaluator bundle declares `accepted_compose_hashes`, the report's compose-hash MUST be in that list.
*Failure mode:* report is forged or vendor-rejected; the report binds a different pubkey than the receipt cached; the hardware doesn't meet the protocol's required configuration (Intel TDX + NVIDIA H100/H200 in TEE mode); the attested compose-hash is not in the publisher's allowlist when one is declared.

**Step 6: Verify the enclave signature.**
Verify `receipt.enclave_signature` over `receipt.response_content` using `receipt.enclave_pubkey`.
*Failure mode:* signature was produced over a different message, or by a different key, or has been tampered with.

**Step 7 (optional, requires provider access): Cross-check via provider.**
Fetch the signature from `receipt.provider_address`'s signature endpoint using `receipt.chat_id`. Confirm it matches `receipt.enclave_signature`. Optionally fetch the provider's published service metadata and confirm the declared `model_identifier` from the evaluator bundle matches the model the provider currently serves at `provider_address`.
*Failure mode:* the cached signature has been swapped post-construction; the provider has rotated keys and the provider-side signature differs; the provider serves a different model than the evaluator declares.

Step 7 is OPTIONAL because it requires the provider to be online and reachable. Steps 1–6 are sufficient for offline verification using only the receipt and Storage.

### 7.2 What a successful verification establishes

On success, a verifier has confirmed:

- The receipt was constructed correctly (Step 1).
- The producer was evaluated under the criteria the receipt names, and those criteria are publicly readable (Steps 2–3).
- The response was produced inside genuine TEE hardware (Steps 4–5).
- The response in the receipt is the same response the TEE signed (Step 6).

What verification does NOT establish:

- That the response is correct or the evaluation is good.
- That the model the TEE loaded was the model the evaluator declared (the TEE attests hardware, not model identity — see §8).
- That the input the producer provided was meaningful, well-formed, or representative of their actual artifact.

---

## 8. Compose vs. Model Identity Binding

EER's most significant residual gap is that the TEE attestation binds the *launched compose configuration*, not the *loaded model weights*. Whether a particular receipt's model claim has any cryptographic backing depends on what that compose configuration actually contains, which varies sharply across providers in practice.

This section describes what TEE attestation does bind on 0G TeeML today, the empirical state of acknowledged providers, and the partial mitigations available — including the protocol-level mitigation EER provides via `accepted_compose_hashes` (§6.5).

### 8.1 What attestation binds

The Intel TDX quote is signed by the platform and binds:

- **MRTD** — the build-time TD measurement (firmware + initrd hashed at TD launch).
- **RTMR0–2** — runtime-extended measurements of the firmware-platform config, the kernel + bootloader, and the kernel cmdline + initramfs scripts. These cover the dstack OS image.
- **RTMR3** — the application-level measurement, extended via the dstack event log. Among the events extended into RTMR3 is `compose-hash`, whose payload equals `sha256(app_compose)`. `app_compose` is a JSON document declaring the docker-compose file, pre-launch script, and runtime config that the TD launches at boot.
- **report_data** — the EVM address of the TEE signer keypair, baked into the quote so verifiers can confirm `enclave_pubkey` originates from this TD.

The compose declaration determines what containers run with what command lines. It MAY include a model identifier as a launch-time argument (for example, vLLM's `--model zai-org/GLM-5-FP8`), in which case RTMR3 cryptographically binds that string. It does NOT measure the actual weight bytes loaded by the model server at runtime — those typically come from a HuggingFace pull or a host-mounted volume that the attestation does not cover.

So the strongest model-identity claim available today, when it holds, is "the launched server was instructed to load this HuggingFace model id." It is not "these weights match a published reference."

### 8.2 Empirical state on 0G TeeML

Reports pulled from all 7 acknowledged mainnet TeeML providers in 2026-04 fall into three categories. Every receipt's strength depends on which category its provider belongs to.

- **Category A — bound launch string** (1 of 7 observed). The attested compose runs the model serving framework directly with the model identifier on the command line. RTMR3 binds the compose-hash, the compose names the model identifier, so the receipt's model claim has cryptographic weight at the granularity of "the launch command named this model." Weights themselves are downloaded at runtime and not measured; image tags may be unpinned (e.g., `vllm/vllm-openai:nightly`).

- **Category B — unrelated compose** (3 of 7 observed). The attested compose runs a generic application image with no model reference at all (in the empirical sample, a Phala demo Next.js starter). The on-chain advertised model and the attested compose are unrelated. Receipts under such providers verify "some dstack TD ran a starter app" — the model claim has no attestation backing.

- **Category C — centralized passthrough** (3 of 7 observed). The attested compose runs a broker proxy whose configuration is mounted from host state outside the attestation. The provider's broker code admits in its API responses that LLM attestation is not available for these services because they route to a centralized API. The TEE attests a proxy; the model serves from a backend with no attestation at all.

This split is empirical, not normative — 0G's protocol does not categorize providers this way. The categorization is a verifier's diagnostic, derived by parsing the attested `app_compose` and observing what it declares.

### 8.3 Mitigations

These are layered. Verifiers and publishers may combine them.

- **Compose-hash allowlist (protocol-level, EER-native).** An evaluator publisher populates `accepted_compose_hashes` (§6.5) with the compose-hashes of providers whose configuration they have inspected and accept. Verification Step 5 fails closed when the attested compose-hash is not in the list. This is the strongest defense the receipt format alone can offer: it shifts trust from "any TEE provider" to "any provider whose compose the publisher reviewed." It is opt-in; bundles without the field do no compose-hash gating.

- **Compose inspection at verification time (advisory).** Even without an allowlist, a verifier can parse the attested `app_compose` and check that the declared `model_identifier` appears in a launch command. This catches Category B and C providers automatically. It is not protocol-mandated because the parsing is provider-specific and brittle, but tooling MAY perform it.

- **Provider service metadata cross-check (advisory; Step 7).** Fetches the provider's on-chain service metadata and confirms it advertises the model the evaluator declares. This catches naming mismatches but cannot detect that the on-chain claim is itself decoupled from the attested compose, so it is weaker than compose inspection.

- **Multi-provider sampling (higher-layer).** A higher-layer protocol can require receipts from multiple independent providers under the same evaluator. Substantial divergence in scores is a signal that one or more providers may not be running the declared model.

- **Provider reputation (higher-layer).** Prefer providers with established track records and verified compose configurations. The 0G Compute Network's acknowledged services list is a starting point but, per §8.2, is not by itself sufficient — being on the list does not imply Category A.

The only protocol-level guarantee is the compose-hash allowlist. The rest are advisory or higher-layer; they compose with EER to tighten the model-environment claim where it matters.

### 8.4 Future protocol primitives

A complete fix for the model-identity gap requires upstream support from the TEE inference protocol. Two paths a verifier-conscious evolution would take:

- **Content-addressed weight binding.** A controlled launcher inside the enclave verifies a weight bundle against a declared content hash and extends the hash into RTMR3 before serving. Closes the runtime-weights gap entirely; requires changes to the inference container.
- **In-enclave weight measurement.** The model server measures weights as they load into memory and extends the result into RTMR3 (or a dedicated runtime measurement register) before accepting requests. Same effect, different implementation point.

When such primitives become available, EER MAY add a `model_measurement` field to the attestation block to carry the weight-level binding without changing other receipt guarantees.

Until then, model-identity binding is partial. The compose-hash allowlist closes the launch-time-string subset of the gap; the runtime-weights subset remains.

---

## 9. Security Considerations

### 9.1 Trust dependencies

EER's security rests on:

- **Intel TDX root certificates and the NVIDIA GPU Attestation API.** A compromise of either vendor's root keys compromises every receipt produced under that vendor's hardware. There is no protocol-level mitigation; this is the cost of hardware-rooted trust.
- **0G Compute Network provider liveness for Step 7 cross-check.** Steps 1–6 are offline; Step 7 is online and optional.
- **Content-addressed storage retrievability.** If an evaluator bundle or attestation report becomes unfetchable, receipts referencing it cannot be verified. EER does not specify storage replication or pinning; producers and verifiers SHOULD ensure relevant artifacts are pinned with sufficient redundancy.

### 9.2 Threat model

**Threat: Evaluator publisher publishes flawed criteria.**
A poorly designed system prompt or scoring schema produces meaningless receipts. EER attests that evaluation happened; it does not attest that the evaluation was good.
*Mitigation:* criteria are public and content-addressed. Anyone can read them before relying on receipts produced under them. Higher-layer evaluator-reputation protocols MAY filter for trusted publishers.

**Threat: Compute provider colludes with producer to produce favorable receipts.**
The provider is honest about TEE setup but cooperates with a producer who submits cherry-picked inputs. The receipts are authentic but the evaluation is rigged.
*Mitigation:* none structurally. EER is per-receipt authenticity, not honest-evaluation. Higher-layer protocols MAY require receipts be produced against committed inputs (`input_commitment`) the producer cannot retroactively choose, or require append-only publication of receipt sequences.

**Threat: Compute provider runs a different model than the evaluator declares.**
See §8 for full discussion. The provider's TEE attests "real hardware ran something"; it does not attest "real hardware ran the model the evaluator named."
*Mitigation:* partial; covered in §8.2.

**Threat: Attestation report forgery.**
An attacker fabricates a report claiming TEE hardware that wasn't used.
*Mitigation:* Step 5's verification of the report against Intel and NVIDIA root certificates. Forgery requires compromising vendor signing keys.

**Threat: Receipt tampering.**
An attacker modifies receipt fields after construction.
*Mitigation:* Step 1's `receipt_id` recomputation detects any change to fields in the canonical signing payload. Step 6's signature verification detects any change to `response_content`. Tampering with `enclave_pubkey`, `enclave_signature`, or `attestation_report_hash` is detected at Steps 5–6 because the signature won't verify against the wrong pubkey, and a different report won't bind the receipt's pubkey.

**Threat: Receipt replay.**
An attacker takes a real receipt and presents it as if it were their own.
*Mitigation:* receipts are not bound to a producer identity at the protocol level. A receipt is a statement about an evaluation, not about a person. Higher layers that bind receipts to producers (e.g., NFT ownership, marketplace registration) provide replay resistance through the binding mechanism, not through EER.

**Threat: Backdating via `created_at`.**
A producer sets `created_at` to a time in the past.
*Mitigation:* none structural — the timestamp is producer-asserted and signed by the producer's choice of value, not the TEE. Verifiers SHOULD compare against external timing signals when temporal claims matter (block timestamps, public-log entries, the producer's first publication of the receipt).

### 9.3 Known gaps

These are limitations of EER v0.5 that future versions MAY address:

- **No model-weight attestation.** TEE attests the launched compose, not the loaded weights. The compose-hash allowlist (§6.5, §8.3) lets a publisher gate on a known-good launch configuration but does not bind the weight bytes that the model server actually loads at runtime. This is a vendor-protocol gap, not a receipt-format gap. See §8.
- **No chain-completeness guarantee.** Receipt chains are producer-asserted. A producer can hide unfavorable receipts. Append-only public logs would address this; v0.5 leaves it to higher layers.
- **No native EVM verification.** Canonical JSON is portable but not native to EVM contracts. EIP-712 typed-data encoding would enable on-chain verification of receipts.
- **No revocation primitive.** A bundle published to Storage cannot be deprecated at the protocol level. Higher-layer registries MAY mark evaluator IDs as deprecated; EER receipts under deprecated IDs remain technically valid but socially deprecated.

---

## 10. Compatibility and extensibility

### 10.1 Protocol extensions

Implementations MAY add fields to `EnhancedReceipt` for protocol-specific extensions, provided that:

- Added fields are namespaced under a top-level `extensions` dict, not added at the receipt's top level alongside core fields. Keys SHOULD be reverse-DNS-style namespace strings (e.g. `"com.example.experiment_id"`) to prevent collisions across publishers.
- The `extensions` field is included in the canonical signing payload, in the position specified by §6.3, and follows the §6.4 canonical-JSON encoding rules at every depth (lexicographic key ordering, no insignificant whitespace, `null` for absent values). Two receipts with byte-identical `extensions` dicts produce the same `receipt_id`.
- Receipts with absent or empty `extensions` encode as `null` in the canonical signing payload (per §6.4's "absent optional fields are `null`, not omission" rule). Implementations SHOULD canonicalize an empty dict `{}` to `null` before signing so "no extensions" has a single canonical form.
- Verifiers MAY ignore extension fields they don't understand; receipts with unknown extensions remain verifiable for their core claims. A verifier MUST NOT reject a receipt solely on the basis of unrecognized extension keys.

### 10.2 Versioning

This document specifies EER v0.5. Receipts under v0.5 carry an explicit `spec_version` field set to `"0.5"` (§6.1). Verifiers MUST refuse receipts whose `spec_version` does not match the version they implement.

**Breaks are asymmetric.** A v0.5 verifier rejects v0.4 receipts (the v0.4 signing payload omits `spec_version`, `evaluator_storage_root`, and `attestation_storage_root`, so the receipt_id digest cannot match under v0.5 rules). A v0.4 verifier presented with a v0.5 receipt would similarly fail Step 1 — it would not know about the new fields and would compute a different digest. The intent is loud incompatibility, not silent forward-compatibility: bumping `spec_version` is the protocol-level signal that older verifiers cannot validate.

**Why the break in v0.5.** v0.4 carried only sha256 content hashes for storage-resident artifacts; verifiers needed an out-of-band sha256→backend-locator index to fetch them. In practice this index lived in producer-side process state, so a verifier with a different storage instance than the producer could not fetch any blob the producer uploaded. v0.5 closes this by adding `evaluator_storage_root` and `attestation_storage_root` to the receipt (§6.1), so any verifier with any backend instance can fetch any receipt's bytes. The break is required: making the new fields optional would let producers ship v0.5 receipts with the bug intact.

**Receipt_ids are not comparable across versions.** v0.4 receipt_ids and v0.5 receipt_ids derived from "the same underlying inference" will differ — the signing payload changed shape. Tooling that indexes by receipt_id MUST treat the version as part of the identity.

Future versions SHOULD continue this convention: `spec_version` is mandatory, in the signing payload, and verifiers refuse mismatching versions.

### 10.3 Substituting protocols

EER's primitives — TEE-attested inference, content-addressed storage — are abstract. While v0.5 is specified against the 0G Compute Network and 0G Storage, equivalent protocols (other TEE inference networks, IPFS, Arweave) MAY be substituted at the implementation level provided they expose:

- Per-call TEE-attested signatures with downloadable attestation reports.
- Content-addressed storage with hash-based retrieval. The storage backend's retrieval locator is what populates `evaluator_storage_root` and `attestation_storage_root` (§6.1); the field name is "storage_root" rather than "zg_root" precisely because the receipt is backend-portable. For 0G Storage the locator is the Merkle rootHash; for IPFS-with-sha256 it is the multihash CID; for Arweave it is the transaction ID.
- A registry of provider identities for `provider_address` resolution.

The receipt schema is unchanged across protocol substitutions, with the exception that backends whose retrieval locators exceed 32 bytes would require widening the storage_root fields from `Bytes32Hex` to `BytesHex` (§6.1). Such a widening is itself a protocol break and would require a corresponding `spec_version` bump.

---

## 11. Out of scope (deferred to future versions)

The following are explicitly NOT specified in v0.5:

- **Cross-receipt aggregation.** Computing reputation, ranking, or composite scores from a set of receipts.
- **Append-only publication.** A canonical log of all receipts produced under a given evaluator or by a given producer.
- **On-chain anchoring.** Periodic publication of receipt-set roots to a blockchain for tamper-evident history.
- **Evaluator governance.** Deprecation, voting, or curation of evaluator bundles.
- **Multi-call evaluations.** A single receipt that spans multiple TEE calls (e.g., chain-of-thought, multi-step grading).
- **Fine-tuning attestation.** Receipts for fine-tuning runs, which have a different attestation flow on 0G Compute.
- **Validator networks.** Independent re-execution and challenge of receipts. EER is a credential protocol, not a consensus protocol.
- **Producer identity binding.** Mapping receipts to NFTs, wallets, or registered producers.
- **Marketplace mechanics.** Pricing, payment, rental, or distribution of evaluated artifacts.
- **Model-measurement binding.** A `model_measurement` field that would close the runtime-weights subset of the §8 binding gap. Pending upstream TEE protocol support; the compose-hash allowlist (§6.5) closes the launch-time-string subset already.

These are valuable extensions that may be specified in future EER versions or in higher-layer protocols built on EER.

---

## 12. Summary

EER converts attested LLM execution into portable, third-party-verifiable evaluation receipts. Public criteria, private input, attested output, durable verification.

The receipt is a small artifact (under 4 KB typical). The verification algorithm is deterministic and offline-capable. The trust model rests on TEE hardware vendors and content-addressed storage. The protocol is general — any LLM evaluation under public criteria fits.

EER is stronger than logs (signed, content-addressed, hardware-rooted), weaker than ZK proofs (trusts hardware vendors), and practical today. Where ZK is unavailable or impractically expensive — most LLM workloads — EER is the strongest available evaluation credential.

This is v0.5. Comments, criticism, and proposed extensions are welcome.

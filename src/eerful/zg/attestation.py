"""Attestation report parsing — dstack TDX quote envelope from 0G TeeML.

The provider's attestation endpoint returns the JSON envelope below. Spec
§7.1 Step 5 needs three things from it: the binding pubkey (from `quote`),
the attested compose-hash (from `tcb_info` + the RTMR3 event log), and
optionally the launched compose itself (for the §8 category diagnostic).

Envelope shape (`GET /v1/quote` against any 0G TeeML provider, verified
against Provider 1 — the GLM-5-FP8 demo target):

```
{
  "quote":          str,    # raw TDX quote, hex-encoded
  "event_log":      str,    # JSON list of dstack event log entries (mirrors tcb_info.event_log)
  "report_data":    str,    # base64; user data hashed into the TDX quote
  "vm_config":      str,    # JSON; dstack VM config
  "tcb_info":       str,    # JSON; structured TCB summary (see ParsedTcbInfo)
  "nvidia_payload": dict,   # NVIDIA Hopper attestation evidence (per-GPU)
}
```

The compose-hash gate (§6.5, §8.3) consumes only `tcb_info`. NVIDIA evidence
and the raw TDX quote are validated by Steps 4–5's vendor chain — out of
scope for this module.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from eerful.canonical import Bytes32Hex, to_lower_hex
from eerful.errors import VerificationError

ComposeCategory = Literal["A", "B", "C", "unknown"]
"""§8.2 categorization of an attested compose. A: bound launch string;
B: unrelated compose; C: centralized passthrough; unknown: heuristics
inconclusive. Diagnostic only — the only protocol-level gate is
`accepted_compose_hashes` (§6.5)."""


class ParsedAttestationReport(BaseModel):
    """The §7.1 Step 5 inputs we extract from a dstack quote envelope.

    Frozen + extra=forbid so callers can't smuggle unverified fields
    through. The raw envelope is intentionally not retained: anything
    Step 5 needs is one of these fields, and dropping the rest keeps
    the surface small.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    compose_hash: Bytes32Hex
    """Lowercase 0x-prefixed sha256(app_compose). Cross-checked against the
    RTMR3 `compose-hash` event payload before construction; the two MUST
    match or the report is malformed."""

    app_compose_raw: str
    """Raw `app_compose` JSON string from `tcb_info`. Hashing this verbatim
    must reproduce `compose_hash` — anything else is a re-serialized copy
    that may differ at the byte level."""

    app_compose: dict[str, Any]
    """Parsed `app_compose`. Used only for the §8 category diagnostic; the
    cryptographic anchor is `compose_hash`, not this dict."""


def _coerce_event_log(raw: Any) -> list[dict[str, Any]]:
    """Accept either a JSON-string-encoded list (top-level `event_log` field)
    or an already-decoded list (`tcb_info.event_log` is sometimes a list,
    sometimes a string depending on the broker version)."""
    if isinstance(raw, str):
        decoded = json.loads(raw)
    else:
        decoded = raw
    if not isinstance(decoded, list):
        raise ValueError(f"event_log is not a list (got {type(decoded).__name__})")
    return decoded


def parse_attestation_report(report_bytes: bytes) -> ParsedAttestationReport:
    """Parse a 0G TeeML attestation envelope and verify the compose-hash chain.

    Mirrors the SDK's `verifyComposeHash` check: confirms
    `sha256(tcb_info.app_compose) == event_log['compose-hash'].event_payload`.
    Without this cross-check, a malicious provider could ship a `tcb_info`
    declaring one compose while RTMR3 attests another.

    Raises `VerificationError(step=5, ...)` for any structural or
    consistency failure — this function is invoked from Step 5 and its
    failures are Step 5 failures.
    """
    try:
        envelope = json.loads(report_bytes)
    except json.JSONDecodeError as e:
        raise VerificationError(step=5, reason=f"attestation report is not valid JSON: {e}") from e

    if not isinstance(envelope, dict):
        raise VerificationError(
            step=5,
            reason=f"attestation envelope is not an object (got {type(envelope).__name__})",
        )

    tcb_info_raw = envelope.get("tcb_info")
    if not isinstance(tcb_info_raw, str):
        raise VerificationError(step=5, reason="attestation envelope missing tcb_info string")

    try:
        tcb_info = json.loads(tcb_info_raw)
    except json.JSONDecodeError as e:
        raise VerificationError(step=5, reason=f"tcb_info is not valid JSON: {e}") from e
    if not isinstance(tcb_info, dict):
        raise VerificationError(
            step=5,
            reason=f"tcb_info is not an object (got {type(tcb_info).__name__})",
        )

    declared_hash_raw = tcb_info.get("compose_hash")
    if not isinstance(declared_hash_raw, str):
        raise VerificationError(step=5, reason="tcb_info missing compose_hash")
    try:
        declared_hash = to_lower_hex(declared_hash_raw)
    except ValueError as e:
        raise VerificationError(
            step=5, reason=f"tcb_info.compose_hash is not hex: {e}"
        ) from e
    # Bytes32 invariant — sha256 is 32 bytes; anything else is a different digest.
    if len(declared_hash) != 66:
        raise VerificationError(
            step=5,
            reason=(
                f"tcb_info.compose_hash is not 32 bytes "
                f"(got {len(declared_hash) - 2} hex chars, expected 64)"
            ),
        )

    app_compose_raw = tcb_info.get("app_compose")
    if not isinstance(app_compose_raw, str):
        raise VerificationError(step=5, reason="tcb_info missing app_compose string")
    try:
        app_compose = json.loads(app_compose_raw)
    except json.JSONDecodeError as e:
        raise VerificationError(step=5, reason=f"app_compose is not valid JSON: {e}") from e
    if not isinstance(app_compose, dict):
        raise VerificationError(step=5, reason="app_compose is not an object")

    # Anchor the declared compose-hash to the actual app_compose bytes. Without
    # this check, a provider can set `tcb_info.compose_hash` and the RTMR3
    # event payload to the same arbitrary value unrelated to `app_compose`,
    # and the categorization heuristic — which reads `app_compose` directly
    # — would be running on data that the TDX quote never bound.
    computed_hash = "0x" + hashlib.sha256(app_compose_raw.encode("utf-8")).hexdigest()
    if computed_hash != declared_hash:
        raise VerificationError(
            step=5,
            reason=(
                f"app_compose hash mismatch: sha256(app_compose) = {computed_hash}, "
                f"tcb_info declares {declared_hash}"
            ),
        )

    # RTMR3 cross-check — without it, tcb_info.compose_hash is just a self-reported
    # field; the event log is the path that actually feeds RTMR3 and gets bound by
    # the TDX quote.
    try:
        event_log = _coerce_event_log(tcb_info.get("event_log"))
    except (ValueError, json.JSONDecodeError) as e:
        raise VerificationError(step=5, reason=f"tcb_info.event_log malformed: {e}") from e

    # Filter on imr==3: events extended into RTMR3 carry imr=3. A compose-hash
    # event recorded under a different IMR is not bound by RTMR3 at all, and
    # accepting it would let a provider record a placeholder compose-hash
    # under (say) imr=0 to satisfy this parser while RTMR3 measures something
    # entirely different. Require exactly one such event so a duplicate or
    # ambiguous binding fails closed.
    compose_events = [
        e
        for e in event_log
        if isinstance(e, dict) and e.get("event") == "compose-hash" and e.get("imr") == 3
    ]
    if not compose_events:
        raise VerificationError(
            step=5,
            reason=(
                "event_log does not contain an RTMR3 'compose-hash' event "
                "(RTMR3 binding missing)"
            ),
        )
    if len(compose_events) > 1:
        raise VerificationError(
            step=5,
            reason="event_log contains multiple RTMR3 'compose-hash' events (binding ambiguous)",
        )
    compose_event = compose_events[0]
    payload_raw = compose_event.get("event_payload")
    if not isinstance(payload_raw, str):
        raise VerificationError(
            step=5,
            reason="compose-hash event missing event_payload",
        )
    try:
        event_hash = to_lower_hex(payload_raw)
    except ValueError as e:
        raise VerificationError(
            step=5,
            reason=f"compose-hash event_payload is not hex: {e}",
        ) from e
    if event_hash != declared_hash:
        raise VerificationError(
            step=5,
            reason=(
                f"compose-hash mismatch: tcb_info declares {declared_hash}, "
                f"RTMR3 event log has {event_hash}"
            ),
        )

    return ParsedAttestationReport(
        compose_hash=declared_hash,
        app_compose_raw=app_compose_raw,
        app_compose=app_compose,
    )


def categorize_compose(
    report: ParsedAttestationReport,
    *,
    expected_model_identifier: str | None = None,
) -> ComposeCategory:
    """Classify an attested compose into a §8.2 category.

    This is a heuristic diagnostic, not a protocol gate. The only normative
    check Step 5 enforces is `accepted_compose_hashes` (§6.5). This function
    powers CLI output ("provider falls in §8 Category A/B/C") so verifiers
    holding receipts without an allowlist can still see the empirical state
    of their provider at a glance.

    Heuristics, in priority order, against the parsed `docker_compose_file`:

    - **Category A** — the bundle's `model_identifier` appears verbatim in
      the compose, or a known model-serving image (`vllm`, `sglang`) is
      named with a `--model` flag. This is the "bound launch string"
      signal: RTMR3's compose-hash binds a string the model identifier
      lives inside.
    - **Category C** — the compose runs a `0g-serving-broker` proxy. The
      broker's own code admits in its 404 response that it routes to
      centralized backends; checking this image name catches that class
      before the weaker Category B check.
    - **Category B** — the compose runs `phala-cloud-nextjs-starter` (the
      empirical Phala demo image observed on Providers 15/16/18) or has no
      model-related strings at all.
    - **unknown** — none of the above signals fired.

    All heuristics are case-insensitive substring matches on the YAML
    serialization of `docker_compose_file`. Brittle by design — providers
    can rename images or restructure the compose. The protocol-level
    answer to "is this Category A?" is "the publisher reviewed it and
    added its hash to `accepted_compose_hashes`".
    """
    raw = report.app_compose.get("docker_compose_file")
    if not isinstance(raw, str):
        return "unknown"
    haystack = raw.lower()

    if expected_model_identifier and expected_model_identifier.lower() in haystack:
        return "A"
    if ("vllm" in haystack or "sglang" in haystack) and "--model" in haystack:
        return "A"
    if "0g-serving-broker" in haystack or "serving-broker" in haystack:
        return "C"
    if "phala-cloud-nextjs-starter" in haystack:
        return "B"
    has_model_keyword = any(k in haystack for k in ("--model", "model_name", "served-model"))
    if not has_model_keyword:
        return "B"
    return "unknown"

"""Attestation report parsing and §8.2 categorization (zg/attestation.py)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from eerful.errors import VerificationError
from eerful.zg.attestation import (
    categorize_compose,
    parse_attestation_report,
)


def _build_report(
    *,
    app_compose: dict[str, Any] | None = None,
    compose_hash_override: str | None = None,
    event_payload_override: str | None = None,
    omit_compose_event: bool = False,
) -> bytes:
    """Build a synthetic attestation report whose RTMR3 event log binds a
    compose-hash matching `tcb_info.compose_hash`.

    The shape mirrors a real `GET /v1/quote` envelope from a 0G TeeML
    provider; only the fields Step 5 reads are populated. Overrides let
    tests force mismatches between the declared hash and the event-log
    hash to exercise the cross-check.
    """
    if app_compose is None:
        app_compose = {
            "manifest_version": 2,
            "name": "test",
            "docker_compose_file": (
                "services:\n"
                "  vllm:\n"
                "    image: vllm/vllm-openai:nightly\n"
                "    command: --model zai-org/GLM-5-FP8 --served-model-name glm-5\n"
            ),
        }
    app_compose_raw = json.dumps(app_compose, sort_keys=True)
    real_hash = hashlib.sha256(app_compose_raw.encode()).hexdigest()
    # Each override is independent: passing only `compose_hash_override` leaves
    # the event_payload at the real hash so the parser's cross-check trips,
    # which is what tests like `test_parse_rejects_compose_hash_mismatch` rely on.
    declared_hash = compose_hash_override if compose_hash_override is not None else real_hash
    event_payload = event_payload_override if event_payload_override is not None else real_hash

    event_log: list[dict[str, Any]] = [
        {"imr": 0, "event_type": 0, "digest": "00" * 48, "event": "boot", "event_payload": ""},
    ]
    if not omit_compose_event:
        event_log.append(
            {
                "imr": 3,
                "event_type": 134217729,
                "digest": "00" * 48,
                "event": "compose-hash",
                "event_payload": event_payload,
            }
        )

    tcb_info = {
        "mrtd": "00" * 48,
        "rtmr3": "00" * 48,
        "compose_hash": declared_hash,
        "event_log": event_log,
        "app_compose": app_compose_raw,
    }

    envelope = {
        "quote": "00" * 16,
        "event_log": json.dumps(event_log),
        "report_data": "",
        "vm_config": "{}",
        "tcb_info": json.dumps(tcb_info),
        "nvidia_payload": {},
    }
    return json.dumps(envelope).encode()


def test_parse_extracts_compose_hash() -> None:
    parsed = parse_attestation_report(_build_report())
    assert parsed.compose_hash.startswith("0x")
    assert len(parsed.compose_hash) == 66
    assert isinstance(parsed.app_compose, dict)


def test_parse_rejects_invalid_json() -> None:
    with pytest.raises(VerificationError) as exc:
        parse_attestation_report(b"not json")
    assert exc.value.step == 5


def test_parse_rejects_missing_tcb_info() -> None:
    with pytest.raises(VerificationError) as exc:
        parse_attestation_report(b'{"quote": "00"}')
    assert exc.value.step == 5
    assert "tcb_info" in exc.value.reason


def test_parse_rejects_rtmr3_event_mismatch() -> None:
    """tcb_info.compose_hash must match the RTMR3 event log entry; otherwise
    the report is internally inconsistent and Step 5 must reject it.

    Forced via `event_payload_override` only — leaving `tcb_info.compose_hash`
    at the real `sha256(app_compose)` so the app_compose cross-check passes
    and we exercise the RTMR3 path specifically."""
    with pytest.raises(VerificationError) as exc:
        parse_attestation_report(_build_report(event_payload_override="ab" * 32))
    assert exc.value.step == 5
    assert "compose-hash mismatch" in exc.value.reason


def test_parse_rejects_app_compose_mutation() -> None:
    """The §7.1 Step 5 anchor is `sha256(app_compose) == declared compose_hash`.
    Without this check, a provider could publish any `app_compose` they like
    and pair it with a fixed declared/event hash, breaking the §8.2
    categorization that reads `app_compose` directly. Mutate the raw
    `app_compose` after construction while leaving both hash fields
    untouched and assert Step 5 fails."""
    valid = _build_report()
    envelope = json.loads(valid)
    tcb = json.loads(envelope["tcb_info"])
    # Swap the bytes of app_compose so its real sha256 no longer matches
    # `compose_hash`. Keep the event_payload aligned with `compose_hash` so
    # the RTMR3 check would still pass — only the new anchor catches this.
    tcb["app_compose"] = json.dumps({"docker_compose_file": "services: {}\n"})
    envelope["tcb_info"] = json.dumps(tcb)

    with pytest.raises(VerificationError) as exc:
        parse_attestation_report(json.dumps(envelope).encode())
    assert exc.value.step == 5
    assert "app_compose hash mismatch" in exc.value.reason


def test_parse_rejects_missing_compose_event() -> None:
    with pytest.raises(VerificationError) as exc:
        parse_attestation_report(_build_report(omit_compose_event=True))
    assert exc.value.step == 5
    assert "compose-hash" in exc.value.reason


def test_parse_rejects_compose_event_under_wrong_imr() -> None:
    """Events extended into RTMR3 carry imr=3. A compose-hash event recorded
    under a different IMR is not bound by RTMR3 and must be ignored —
    otherwise a provider could plant a placeholder under (say) imr=0 to
    satisfy this parser while RTMR3 measures something entirely different."""
    valid = _build_report()
    envelope = json.loads(valid)
    tcb = json.loads(envelope["tcb_info"])
    for e in tcb["event_log"]:
        if e.get("event") == "compose-hash":
            e["imr"] = 0
    envelope["tcb_info"] = json.dumps(tcb)
    with pytest.raises(VerificationError) as exc:
        parse_attestation_report(json.dumps(envelope).encode())
    assert exc.value.step == 5
    assert "RTMR3 'compose-hash' event" in exc.value.reason


def test_parse_rejects_multiple_compose_events() -> None:
    """Two RTMR3 compose-hash events make the binding ambiguous; fail closed."""
    valid = _build_report()
    envelope = json.loads(valid)
    tcb = json.loads(envelope["tcb_info"])
    compose_event = next(e for e in tcb["event_log"] if e.get("event") == "compose-hash")
    tcb["event_log"].append({**compose_event})  # duplicate
    envelope["tcb_info"] = json.dumps(tcb)
    with pytest.raises(VerificationError) as exc:
        parse_attestation_report(json.dumps(envelope).encode())
    assert exc.value.step == 5
    assert "multiple" in exc.value.reason or "ambiguous" in exc.value.reason


def test_parse_rejects_non_dict_tcb_info() -> None:
    """`tcb_info` must be a JSON object. A list or string parses as valid JSON
    but `.get(...)` would crash with AttributeError; the parser must
    raise `VerificationError(step=5)` instead so callers see structured
    failures."""
    envelope = {
        "quote": "00",
        "event_log": "[]",
        "report_data": "",
        "vm_config": "{}",
        "tcb_info": json.dumps(["not", "a", "dict"]),
        "nvidia_payload": {},
    }
    with pytest.raises(VerificationError) as exc:
        parse_attestation_report(json.dumps(envelope).encode())
    assert exc.value.step == 5
    assert "tcb_info" in exc.value.reason


def test_parse_normalizes_uppercase_compose_hash() -> None:
    """The bridge / provider may serialize the compose-hash uppercase; spec
    §6.4 forces lowercase. The parser MUST canonicalize on the way in or
    `accepted_compose_hashes` membership checks will spuriously fail.

    Mutates the envelope post-construction so the declared/event hashes go
    uppercase but still match `sha256(app_compose)` byte-for-byte."""
    valid = _build_report()
    envelope = json.loads(valid)
    tcb = json.loads(envelope["tcb_info"])
    real_hash = tcb["compose_hash"]
    upper = real_hash.upper()
    tcb["compose_hash"] = upper
    for e in tcb["event_log"]:
        if e.get("event") == "compose-hash":
            e["event_payload"] = upper
    envelope["tcb_info"] = json.dumps(tcb)

    parsed = parse_attestation_report(json.dumps(envelope).encode())
    assert parsed.compose_hash == "0x" + real_hash


def test_categorize_a_via_model_identifier() -> None:
    parsed = parse_attestation_report(_build_report())
    assert categorize_compose(parsed, expected_model_identifier="zai-org/GLM-5-FP8") == "A"


def test_categorize_a_via_vllm_model_flag() -> None:
    """A compose can be Category A even if the bundle's model_identifier
    isn't a verbatim substring — `vllm` + `--model` is enough signal."""
    parsed = parse_attestation_report(_build_report())
    assert categorize_compose(parsed, expected_model_identifier="other/model") == "A"


def test_categorize_b_phala_starter() -> None:
    parsed = parse_attestation_report(
        _build_report(
            app_compose={
                "docker_compose_file": (
                    "services:\n"
                    "  app:\n"
                    "    image: leechael/phala-cloud-nextjs-starter:latest\n"
                ),
            }
        )
    )
    assert categorize_compose(parsed, expected_model_identifier="anything") == "B"


def test_categorize_c_serving_broker() -> None:
    parsed = parse_attestation_report(
        _build_report(
            app_compose={
                "docker_compose_file": (
                    "services:\n"
                    "  broker:\n"
                    "    image: ghcr.io/0gfoundation/0g-serving-broker:latest\n"
                ),
            }
        )
    )
    assert categorize_compose(parsed, expected_model_identifier="anything") == "C"


def test_categorize_unknown_when_model_keyword_present_but_unrecognized() -> None:
    parsed = parse_attestation_report(
        _build_report(
            app_compose={
                "docker_compose_file": (
                    "services:\n"
                    "  worker:\n"
                    "    image: example/custom:latest\n"
                    "    command: --model-name foo\n"
                ),
            }
        )
    )
    assert categorize_compose(parsed, expected_model_identifier=None) == "unknown"

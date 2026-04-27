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


def test_parse_rejects_compose_hash_mismatch() -> None:
    """tcb_info.compose_hash must match the RTMR3 event log entry; otherwise
    the report is internally inconsistent and Step 5 must reject it."""
    bogus = "ab" * 32
    with pytest.raises(VerificationError) as exc:
        parse_attestation_report(_build_report(compose_hash_override=bogus))
    assert exc.value.step == 5
    assert "compose-hash mismatch" in exc.value.reason


def test_parse_rejects_missing_compose_event() -> None:
    with pytest.raises(VerificationError) as exc:
        parse_attestation_report(_build_report(omit_compose_event=True))
    assert exc.value.step == 5
    assert "compose-hash" in exc.value.reason


def test_parse_normalizes_uppercase_compose_hash() -> None:
    """The bridge / provider may serialize the compose-hash uppercase; spec
    §6.4 forces lowercase. The parser MUST canonicalize on the way in or
    `accepted_compose_hashes` membership checks will spuriously fail."""
    h_lower = "ab" * 32
    h_upper = h_lower.upper()
    parsed = parse_attestation_report(
        _build_report(compose_hash_override=h_upper, event_payload_override=h_upper)
    )
    assert parsed.compose_hash == "0x" + h_lower


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

"""Exception hierarchy for eerful.

Verifiers raise `VerificationError` with a step number so callers can attribute
failures to specific algorithm steps in spec §7. `TrustViolation` is reserved
for situations where a remote party returned data that does not match what the
content hash promised — these should never recover silently.
"""

from __future__ import annotations


class EerError(Exception):
    """Base for everything raised by eerful."""


class VerificationError(EerError):
    """A receipt failed verification at a specific step (spec §7)."""

    def __init__(self, step: int, reason: str) -> None:
        self.step = step
        self.reason = reason
        super().__init__(f"verification step {step}: {reason}")


class TrustViolation(EerError):
    """A remote party returned data that does not match its content hash.

    Never recover from this silently — it indicates either a bug, a misconfigured
    storage backend, or an active substitution attempt.
    """


class StorageError(EerError):
    """0G Storage operation failed (network, gas, indexing, etc.)."""


class ComputeError(EerError):
    """0G Compute (TeeML) operation failed."""


class EvaluationClientError(EerError):
    """Caller passed parameters to EvaluationClient that conflict with the bound
    evaluator bundle's criteria (e.g. overriding system prompt or temperature)."""

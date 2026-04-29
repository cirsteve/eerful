"""eerful — reference implementation of Enhanced Evaluation Receipts.

Spec: docs/spec.md (canonical).
"""

from eerful.canonical import (
    Address,
    Bytes32Hex,
    BytesHex,
    canonical_json_bytes,
    is_address,
    is_bytes32_hex,
    is_bytes_hex,
    to_lower_hex,
)
from eerful.commitment import SaltStore, compute_input_commitment, generate_salt
from eerful.errors import (
    ComputeError,
    EerError,
    EvaluationClientError,
    StorageError,
    TrustViolation,
    VerificationError,
)
from eerful.evaluator import EvaluatorBundle
from eerful.receipt import EnhancedReceipt

__version__ = "0.1.0"

__all__ = [
    "Address",
    "Bytes32Hex",
    "BytesHex",
    "ComputeError",
    "EerError",
    "EnhancedReceipt",
    "EvaluationClientError",
    "EvaluatorBundle",
    "SaltStore",
    "StorageError",
    "TrustViolation",
    "VerificationError",
    "__version__",
    "canonical_json_bytes",
    "compute_input_commitment",
    "generate_salt",
    "is_address",
    "is_bytes32_hex",
    "is_bytes_hex",
    "to_lower_hex",
]

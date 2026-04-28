"""jig adapter for eerful — produce EER receipts during pipeline grading.

Three pieces compose:

- `EvaluationClient(LLMClient)` — a jig LLM client that runs every
  `.complete()` against a TEE provider and attaches the resulting
  `EnhancedReceipt` to the response.
- `EvaluationGrader(Grader[Any])` — a jig grader backed by an
  `EvaluationClient`. Calls the client, parses the bundle's
  `output_score_block`, returns one `Score` per numeric dimension, and
  optionally persists the receipt via a `FeedbackLoop`.
- `attach_receipt_to_span(span, receipt)` — span-decoration helper that
  any tracer can use; not its own `TracingLogger` so users keep their
  existing tracer choice.

Read order for understanding: client → grader → tracer. The grader is
where the jig pipeline meets the EER receipt; the client owns the
TeeML call mechanics; the tracer is a one-function helper.
"""

from eerful.jig.client import (
    EerfulLLMResponse,
    EvaluationClient,
)
from eerful.jig.grader import EvaluationGrader
from eerful.jig.tracer import attach_receipt_to_span

__all__ = [
    "EerfulLLMResponse",
    "EvaluationClient",
    "EvaluationGrader",
    "attach_receipt_to_span",
]

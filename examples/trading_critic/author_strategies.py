"""author_strategies — maintainer tool, NOT in demo flow.

Drafts `strategies/v2.md` and `strategies/v3.md` by asking Claude to
evolve the prior version along specific axes designed for the
trading-critic's scoring rubric.

Workflow:

1. `python author_strategies.py --to v2` — reads strategies/v1.md,
   writes strategies/v2.md (a draft adding stop-loss + ATR-based
   position sizing).
2. Hand-edit `strategies/v2.md` until the language is clean.
3. `python author_strategies.py --to v3` — reads strategies/v2.md,
   writes strategies/v3.md (a draft adding regime detection +
   contrarian sub-strategy).
4. Hand-edit `strategies/v3.md`.
5. `python demo.py` (eventually) to see the critic's actual scores
   on the strategy chain.

Refuses to overwrite an existing file unless `--force` is given. v2
and v3 are gitignored, so re-running this script is a local-only
operation and does not affect committed history.

Requires `ANTHROPIC_API_KEY` in the environment (or in the repo-root
`.env`). The `anthropic` SDK is available transitively via the
`jig[anthropic]` dep already in `pyproject.toml`.

NOT a runtime dependency for `demo.py` or `verify_chain.py`. The demo
treats v1/v2/v3 as on-disk files and never reaches for an API.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anthropic

_HERE = Path(__file__).resolve().parent
_STRATEGIES = _HERE / "strategies"

_DEFAULT_MODEL = "claude-opus-4-7"

# Each version's prompt names the specific axes the critic is going to
# score on, so the draft ships with material the critic can engage with.
# Hand-tuning these prompts is the highest-leverage thing in this file —
# they are the difference between scores that show monotonic improvement
# and scores that bunch in the middle.
_EVOLUTION_PROMPTS: dict[str, str] = {
    "v2": (
        "You are drafting v2 of a trading strategy by evolving v1 along two "
        "specific axes:\n\n"
        "1. ADD a defensive risk overlay. Spell out a concrete stop-loss "
        "policy (e.g., a fixed % below entry, or a trailing stop tied to "
        "ATR), and describe how it interacts with the existing crossover "
        "exits.\n"
        "2. REPLACE the 100%-of-capital sizing with volatility-aware "
        "position sizing (e.g., target a fixed dollar-volatility per "
        "position via 14-day ATR). Show the math.\n\n"
        "DO NOT add regime detection, contrarian logic, or a multi-asset "
        "universe. Save those for v3. Keep the markdown structure and "
        "tone of v1; this should read like the same author wrote both."
    ),
    "v3": (
        "You are drafting v3 of a trading strategy by evolving v2 along two "
        "specific axes:\n\n"
        "1. ADD a regime filter. Describe a concrete signal (e.g., 200-day "
        "MA slope, realized-vol percentile) that classifies the current "
        "market into a small number of regimes, and how the strategy "
        "behaves differently in each.\n"
        "2. ADD a contrarian sub-strategy that activates in the "
        "low-volatility / mean-reverting regime — a brief mean-reversion "
        "rule (e.g., short-term RSI extremes) that complements the "
        "trend-following crossover. Specify how the two sub-strategies "
        "share capital.\n\n"
        "Keep v2's risk overlay and sizing intact. Keep the markdown "
        "structure and tone of v2."
    ),
}

_SYSTEM_PROMPT = (
    "You write concise trading-strategy markdown documents in the voice "
    "of a thoughtful retail quant. You produce ONLY the markdown body "
    "of the strategy document — no preamble, no closing remarks, no "
    "explanation outside the document. The document follows the same "
    "section structure as its predecessor."
)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _seed_for(target: str) -> Path:
    """Return the prior version's path that this target evolves from."""
    return _STRATEGIES / {"v2": "v1.md", "v3": "v2.md"}[target]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="author_strategies")
    parser.add_argument(
        "--to",
        required=True,
        choices=("v2", "v3"),
        help="which strategy version to draft",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the target file if it already exists",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Claude model to author with (default: {_DEFAULT_MODEL})",
    )
    args = parser.parse_args(argv)

    repo_root = _HERE.parent.parent
    _load_dotenv(repo_root / ".env")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY not set. Add it to eerful/.env or your shell.",
            file=sys.stderr,
        )
        return 2

    target_path = _STRATEGIES / f"{args.to}.md"
    if target_path.exists() and not args.force:
        print(
            f"{target_path} already exists. Pass --force to overwrite, "
            "or hand-edit the existing file directly.",
            file=sys.stderr,
        )
        return 2

    seed_path = _seed_for(args.to)
    if not seed_path.exists():
        print(f"seed strategy {seed_path} does not exist", file=sys.stderr)
        return 2

    seed_text = seed_path.read_text()
    user_prompt = (
        f"Here is strategy {seed_path.stem}.md:\n\n"
        f"{seed_text}\n\n"
        f"---\n\n"
        f"{_EVOLUTION_PROMPTS[args.to]}\n\n"
        f"Write the {args.to} markdown document now."
    )

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=args.model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Extract the text content. The SDK returns a list of content blocks;
    # for our prompt shape (no tools, plain text) there is one TextBlock.
    # An empty list means the model returned only non-text blocks (a
    # refusal, a stop-on-max-tokens with no output, a future content
    # type we don't recognize) — silently writing the resulting "\n"
    # would clobber an existing v2/v3 file with a blank draft and
    # masquerade as success. Fail loudly with the stop_reason instead.
    parts = [block.text for block in message.content if block.type == "text"]
    if not parts:
        raise RuntimeError(
            f"authoring model returned no text blocks for {args.to} "
            f"(stop_reason={message.stop_reason!r}); "
            f"target file {target_path} not written"
        )
    body = "\n\n".join(parts).strip() + "\n"

    target_path.write_text(body)
    print(f"wrote {target_path} ({len(body)} chars)")
    print("Hand-edit before running demo.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

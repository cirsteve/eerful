"""bundle_inspect — maintainer tool for the trading-critic bundle.

Two modes; both are maintainer-only and NOT part of the demo flow.

**Inspect mode** (default): fetches Provider 1's current attestation
report, computes the live compose-hash, and compares it against the
value pinned in `bundle.json`'s `accepted_compose_hashes` allowlist.
Surfaces the §8.2 category for sanity (Provider 1 is the only
Category A provider observed, so a drop to Category B/C indicates the
provider rotated to a different compose).

Use cases:

- Before publishing a fresh bundle: run `python bundle_inspect.py
  --confirm-compose-hash` and paste the printed hash into
  `bundle.json` under `accepted_compose_hashes`.
- Morning of submission: run plain `python bundle_inspect.py` to
  confirm the pinned hash still matches Provider 1's current attestation.
  A mismatch means re-author + re-record receipts.

**Score-test mode** (`--score-test`): runs the critic against
`strategies/v1.md`, `v2.md`, `v3.md` and prints scores side-by-side.
Doesn't produce receipts — this is the prompt-iteration tool the
maintainer uses while tuning `bundle.json`'s system_prompt for
monotonic improvement across the chain. Costs three real broker
inference calls per run.

Bridge prerequisite for both modes: the local zg-bridge must be
running. See `services/zg-bridge/README.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from eerful.errors import ComputeError
from eerful.evaluator import EvaluatorBundle
from eerful.zg.attestation import categorize_compose, parse_attestation_report
from eerful.zg.bridge_init import bridge_init
from eerful.zg.compute import ComputeClient


_DEFAULT_BUNDLE = Path(__file__).resolve().parent / "bundle.json"
_DEFAULT_PROVIDER = "0xd9966e13a6026Fcca4b13E7ff95c94DE268C471C"
_STRATEGIES_DIR = Path(__file__).resolve().parent / "strategies"
_VERSIONS: tuple[str, str, str] = ("v1", "v2", "v3")
_SCORE_DIMS: tuple[str, ...] = ("risk", "novelty", "robustness", "overall")


def _load_dotenv(path: Path) -> None:
    """Tiny KEY=VALUE .env loader — mirrors `examples/smoke_testnet.py`."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _run_score_test(
    *,
    bundle: EvaluatorBundle,
    provider_address: str,
    bridge_url: str,
    as_json: bool,
    verbose: bool,
) -> int:
    """Run the critic against v1/v2/v3 and print scores side-by-side.

    No receipt construction, no storage uploads — just the critic's
    raw response per strategy, parsed for its score block. Used by
    the maintainer's prompt-hardening loop to find the
    `system_prompt + strategies/*.md` combination that produces
    clearly differentiated, monotonically-improving scores before
    committing to a real receipt chain.

    Three broker inference calls per run; not free. The bundle's
    `inference_params` are honored so the result reflects what the
    real demo would produce.
    """
    missing = [
        str(_STRATEGIES_DIR / f"{v}.md")
        for v in _VERSIONS
        if not (_STRATEGIES_DIR / f"{v}.md").exists()
    ]
    if missing:
        print(
            "  ✗ missing strategy files (v2/v3 are gitignored — draft them "
            "with author_strategies.py):\n      "
            + "\n      ".join(missing),
            file=sys.stderr,
        )
        return 2

    bundle_params = bundle.inference_params or {}
    temperature = bundle_params.get("temperature")
    max_tokens = bundle_params.get("max_tokens")

    results: dict[str, dict[str, Any]] = {}
    with ComputeClient(bridge_url=bridge_url) as compute:
        try:
            status = bridge_init(compute, provider_address)
        except ComputeError as e:
            print(f"  ✗ {e}", file=sys.stderr)
            return 2
        if not as_json:
            print(f"  bridge wallet={status.wallet} chain={status.chain_id}")
            print(
                f"  ledger: created={status.ledger_created} "
                f"balance={status.total_balance_0g:.4f} 0G"
            )

        for version in _VERSIONS:
            strategy_text = (_STRATEGIES_DIR / f"{version}.md").read_text()
            messages = [
                {"role": "system", "content": bundle.system_prompt},
                {"role": "user", "content": strategy_text},
            ]
            # Use `infer`, not `infer_full`. Score-test only needs the
            # `response_content` for JSON parsing — the signature +
            # attestation round-trips that `infer_full` adds are dead
            # weight here (no receipt is being constructed). Saves 2
            # network calls per strategy × 3 strategies = 6 calls per
            # score-test run, which matters when the maintainer is
            # iterating on the system prompt.
            try:
                response = compute.infer(
                    provider_address=provider_address,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except ComputeError as e:
                # Surface which strategy failed so the maintainer
                # knows where to resume — receipts aren't being
                # produced here so we can fail-and-bail rather than
                # try to recover state.
                print(f"  ✗ {version}: inference failed: {e}", file=sys.stderr)
                return 1
            response_content = response.get("response_content", "")
            try:
                parsed = json.loads(response_content)
            except json.JSONDecodeError as e:
                print(
                    f"  ✗ {version}: critic response is not valid JSON: {e}\n"
                    f"      raw response: {response_content!r}",
                    file=sys.stderr,
                )
                return 1
            if not isinstance(parsed, dict):
                print(
                    f"  ✗ {version}: critic response parsed as "
                    f"{type(parsed).__name__}, not a dict.\n"
                    f"      raw response: {response_content!r}",
                    file=sys.stderr,
                )
                return 1
            results[version] = parsed

    if as_json:
        print(json.dumps(results, indent=2))
        return 0

    # Aligned table — column-widths chosen so dimension headers (4-9
    # chars) and 0.000-format values (5 chars) all fit.
    header = "          " + "".join(f"{dim:<10}" for dim in _SCORE_DIMS)
    print()
    print(header)
    for version in _VERSIONS:
        scores = results[version]
        row = f"  {version:<8}"
        for dim in _SCORE_DIMS:
            v = scores.get(dim)
            row += f"{v:<10.3f}" if isinstance(v, (int, float)) else f"{'?':<10}"
        print(row)
    if verbose:
        print()
        for version in _VERSIONS:
            commentary = results[version].get("commentary", "")
            print(f"  {version} commentary: {commentary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Load .env BEFORE constructing the parser. argparse's `default=`
    # values are evaluated at `add_argument` time, so reading
    # `os.environ.get(...)` after `parse_args` would pick up a stale
    # process environment and ignore values set in the repo .env —
    # the publish workflow could then inspect the wrong provider and
    # pin the wrong compose-hash into bundle.json.
    repo_root = Path(__file__).resolve().parent.parent.parent
    _load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser(prog="bundle_inspect")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=_DEFAULT_BUNDLE,
        help="path to the bundle JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=os.environ.get("EERFUL_0G_COMPUTE_PROVIDER_ADDRESS", _DEFAULT_PROVIDER),
        help=(
            "0G compute provider address (default: $EERFUL_0G_COMPUTE_PROVIDER_ADDRESS "
            f"or {_DEFAULT_PROVIDER})"
        ),
    )
    parser.add_argument(
        "--bridge-url",
        type=str,
        default=os.environ.get(
            "EERFUL_0G_BRIDGE_URL",
            f"http://127.0.0.1:{os.environ.get('EERFUL_0G_BRIDGE_PORT', '7878')}",
        ),
        help="bridge URL (default: $EERFUL_0G_BRIDGE_URL or http://127.0.0.1:7878)",
    )
    # Inspect-mode and score-test are alternative workflows that share
    # the same wiring (bundle, provider, bridge, env). Mutually
    # exclusive to keep the help output and exit-code semantics
    # straightforward — the user picks one. confirm-compose-hash is
    # an inspect-mode shortcut, not a third mode.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--confirm-compose-hash",
        action="store_true",
        help=(
            "inspect-mode shortcut: print the live compose-hash on a "
            "single line and exit 0. Convenience for the publish "
            "workflow: paste into bundle.json's accepted_compose_hashes."
        ),
    )
    mode_group.add_argument(
        "--score-test",
        action="store_true",
        help=(
            "run the critic against strategies/v1.md, v2.md, v3.md and "
            "print scores side-by-side. Maintainer's prompt-hardening "
            "loop tool — does NOT produce receipts. Three broker "
            "inference calls per run."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of human-formatted output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "score-test only: also print each strategy's critic "
            "commentary alongside its scores"
        ),
    )
    args = parser.parse_args(argv)

    # Enforce the score-test-only contract on `--verbose` at parse time.
    # Without this, a user passing `--verbose --confirm-compose-hash`
    # would have the flag silently dropped — the help text would say
    # one thing and the script would do another.
    if args.verbose and not args.score_test:
        parser.error("--verbose is only valid with --score-test")

    try:
        bundle = EvaluatorBundle.model_validate_json(args.bundle.read_bytes())
    except OSError as e:
        print(f"failed to read bundle at {args.bundle}: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"bundle does not validate: {e}", file=sys.stderr)
        return 1

    # Score-test mode runs an entirely different flow against the
    # critic; route here before the inspect-mode attestation fetch.
    if args.score_test:
        return _run_score_test(
            bundle=bundle,
            provider_address=args.provider,
            bridge_url=args.bridge_url,
            as_json=args.json,
            verbose=args.verbose,
        )

    # Keep `None` distinct from a populated list. Per spec §6.5 / the
    # bundle validator, `None` means "no allowlist, gating skipped by
    # design"; a populated list means "gate Step 5 on these hashes".
    # Empty lists are structurally impossible (bundle validator rejects).
    declared = bundle.accepted_compose_hashes

    with ComputeClient(bridge_url=args.bridge_url) as client:
        report_bytes, report_hash = client.fetch_attestation(args.provider)

    parsed = parse_attestation_report(report_bytes)
    live_hash = parsed.compose_hash
    category = categorize_compose(parsed, expected_model_identifier=bundle.model_identifier)

    if args.confirm_compose_hash:
        print(live_hash)
        return 0

    # Tri-state gating: distinguishes "no allowlist by design" (skipped,
    # success) from "allowlist exists, live hash isn't in it" (mismatch,
    # failure). Keeps the human and JSON paths' exit codes in lockstep.
    if declared is None:
        gating = "skipped"
    elif live_hash in declared:
        gating = "enforced"
    else:
        gating = "mismatch"

    if args.json:
        print(
            json.dumps(
                {
                    "bundle_path": str(args.bundle),
                    "provider": args.provider,
                    "model_identifier": bundle.model_identifier,
                    "live_compose_hash": live_hash,
                    "declared_compose_hashes": declared,
                    "gating": gating,
                    "category": category,
                    "report_hash": report_hash,
                },
                indent=2,
            )
        )
        return 1 if gating == "mismatch" else 0

    print(f"  bundle: {bundle.version} ({bundle.model_identifier})")
    print(f"  provider: {args.provider}")
    print(f"  live compose-hash: {live_hash}")
    print(f"  §8.2 category: {category}")
    if gating == "skipped":
        print("  declared accepted_compose_hashes: none")
        print("  → bundle has no allowlist; Step 5 gating would be skipped")
        return 0
    assert declared is not None  # narrowed by gating != "skipped"
    if gating == "enforced":
        print(f"  declared accepted_compose_hashes: {len(declared)} entries (LIVE HASH IS PINNED)")
        print("  → bundle is current; receipts will pass Step 5 gating")
        return 0
    # gating == "mismatch"
    print(f"  declared accepted_compose_hashes ({len(declared)} entries):")
    for h in declared:
        print(f"    - {h}")
    print()
    print("  ✗ MISMATCH — Provider 1's compose has rotated since the bundle was pinned.")
    print("  Action: re-author + republish bundle, then regenerate the receipt chain.")
    print(f"           (paste {live_hash} into bundle.json accepted_compose_hashes)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""bundle_inspect — maintainer tool for the trading-critic bundle.

Fetches Provider 1's current attestation report, computes the live
compose-hash, and compares it against the value pinned in `bundle.json`'s
`accepted_compose_hashes` allowlist. Surfaces the §8.2 category for
sanity (Provider 1 is the only Category A provider observed, so a drop
to Category B/C indicates the provider rotated to a different compose).

Use cases:

- Before publishing a fresh bundle: run `python bundle_inspect.py
  --confirm-compose-hash` and paste the printed hash into
  `bundle.json` under `accepted_compose_hashes`.
- Morning of submission: run plain `python bundle_inspect.py` to
  confirm the pinned hash still matches Provider 1's current attestation.
  A mismatch means re-author + re-record receipts.

NOT in the demo flow. demo.py never calls this — the bundle is treated
as an immutable artifact at run time.

Bridge prerequisite: the local zg-bridge must be running. See
`services/zg-bridge/README.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from eerful.evaluator import EvaluatorBundle
from eerful.zg.attestation import categorize_compose, parse_attestation_report
from eerful.zg.compute import ComputeClient


_DEFAULT_BUNDLE = Path(__file__).resolve().parent / "bundle.json"
_DEFAULT_PROVIDER = "0xd9966e13a6026Fcca4b13E7ff95c94DE268C471C"


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
    parser.add_argument(
        "--confirm-compose-hash",
        action="store_true",
        help=(
            "print the live compose-hash on a single line and exit 0. "
            "Convenience for the publish workflow: paste into bundle.json's "
            "accepted_compose_hashes."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of human-formatted output",
    )
    args = parser.parse_args(argv)

    try:
        bundle = EvaluatorBundle.model_validate_json(args.bundle.read_bytes())
    except OSError as e:
        print(f"failed to read bundle at {args.bundle}: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"bundle does not validate: {e}", file=sys.stderr)
        return 1

    declared = bundle.accepted_compose_hashes or []

    with ComputeClient(bridge_url=args.bridge_url) as client:
        report_bytes, report_hash = client.fetch_attestation(args.provider)

    parsed = parse_attestation_report(report_bytes)
    live_hash = parsed.compose_hash
    category = categorize_compose(parsed, expected_model_identifier=bundle.model_identifier)

    if args.confirm_compose_hash:
        print(live_hash)
        return 0

    matches = live_hash in declared
    if args.json:
        print(
            json.dumps(
                {
                    "bundle_path": str(args.bundle),
                    "provider": args.provider,
                    "model_identifier": bundle.model_identifier,
                    "live_compose_hash": live_hash,
                    "declared_compose_hashes": list(declared),
                    "matches": matches,
                    "category": category,
                    "report_hash": report_hash,
                },
                indent=2,
            )
        )
        return 0 if matches else 1

    print(f"  bundle: {bundle.version} ({bundle.model_identifier})")
    print(f"  provider: {args.provider}")
    print(f"  live compose-hash: {live_hash}")
    print(f"  §8.2 category: {category}")
    if not declared:
        print("  declared accepted_compose_hashes: none")
        print("  → bundle has no allowlist; Step 5 gating would be skipped")
        return 0
    if matches:
        print(f"  declared accepted_compose_hashes: {len(declared)} entries (LIVE HASH IS PINNED)")
        print("  → bundle is current; receipts will pass Step 5 gating")
        return 0
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

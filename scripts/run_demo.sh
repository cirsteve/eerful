#!/usr/bin/env bash
# run_demo.sh — single-command, narratable AXL multi-agent demo.
#
# Walks the clean path (both gates PASS) then the poisoned path
# (proposal_grade REFUSES, implementation_grade PASSES — the two-gate
# point). Designed for live screen recording on otto.
#
# Topology assumed up before this runs:
#   - gil   : AXL node on :9001, bridge on :9002    (ssh gil)
#   - louie : AXL node on :9001, refiner.py daemon  (ssh louie)
#   - otto  : zg-bridge on 127.0.0.1:7878
#             ssh tunnel: localhost:9002 → gil:9002
#
# This script preflights all four endpoints and prints the bring-up
# commands if any are missing. See examples/trading/axl/README.md for
# one-time deploy + per-run bring-up.
#
# Usage:
#   scripts/run_demo.sh                # interactive: pauses between scenes
#   scripts/run_demo.sh --no-pause     # run straight through
#   scripts/run_demo.sh --check-only   # preflight only, then exit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PAUSE=true
CHECK_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --no-pause)   PAUSE=false ;;
        --check-only) CHECK_ONLY=true ;;
        -h|--help)
            sed -n '2,21p' "$0"
            exit 0
            ;;
    esac
done

if [ -t 1 ]; then
    BOLD=$(tput bold); DIM=$(tput dim)
    RED=$(tput setaf 1); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
    RESET=$(tput sgr0)
else
    BOLD=; DIM=; RED=; GREEN=; YELLOW=; CYAN=; RESET=
fi

scene() {
    echo
    echo "${BOLD}${CYAN}━━━ $1 ━━━${RESET}"
    echo
}
note()  { echo "${DIM}$*${RESET}"; }
fail()  { echo "${RED}${BOLD}$*${RESET}"; }
ok()    { echo "${GREEN}$*${RESET}"; }

pause() {
    if $PAUSE; then
        echo
        echo "${DIM}[press enter to continue]${RESET}"
        read -r _
    fi
}

run() {
    echo "${YELLOW}\$ $*${RESET}"
    "$@"
}

show_json() {
    if command -v jq >/dev/null 2>&1; then jq . "$1"; else cat "$1"; fi
}

# ─────────────────────────────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────────────────────────────
scene "preflight: topology"

PREFLIGHT_OK=true

# 1. 0G bridge on otto loopback (mints receipts via TEE)
if curl -sf --max-time 3 http://127.0.0.1:7878/healthz >/dev/null 2>&1; then
    ok "  zg-bridge       127.0.0.1:7878   OK"
else
    fail "  zg-bridge       127.0.0.1:7878   DOWN"
    echo "      bring up: cd services/zg-bridge && npm start &"
    PREFLIGHT_OK=false
fi

# 2. AXL bridge via ssh tunnel to gil. /recv returns 204 on empty queue;
#    that's our liveness probe (doesn't consume any queued envelope).
AXL_PROBE_RC=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:9002/recv 2>/dev/null || echo 000)
if [ "$AXL_PROBE_RC" = "204" ] || [ "$AXL_PROBE_RC" = "200" ]; then
    ok "  AXL tunnel→gil  127.0.0.1:9002   OK (HTTP $AXL_PROBE_RC)"
else
    fail "  AXL tunnel→gil  127.0.0.1:9002   DOWN (HTTP $AXL_PROBE_RC)"
    echo "      bring up: ssh -L 9002:127.0.0.1:9002 -N gil &"
    echo "      (assumes AXL nodes are running on gil + louie — see examples/trading/axl/README.md)"
    PREFLIGHT_OK=false
fi

# 3. Refiner on louie — best-effort liveness via ssh. Non-fatal if ssh
#    isn't available (e.g. running on a machine without louie auth);
#    the demo will still surface a refiner-down failure when the AXL
#    round trip times out.
if command -v ssh >/dev/null 2>&1 && \
   ssh -o ConnectTimeout=3 -o BatchMode=yes louie 'pgrep -f refiner.py >/dev/null' 2>/dev/null; then
    ok "  refiner@louie                    OK"
else
    note "  refiner@louie                    (couldn't verify via ssh — proceeding)"
    note "      if down: ssh louie \"cd ~/eerful-axl && AXL_EXPLORER_PEER_ID=<gil-pubkey> nohup ./.venv/bin/python refiner.py > refiner.log 2>&1 &\""
fi

# 4. Required env (refiner peer override is optional; provider is required)
if [ -z "${EERFUL_0G_COMPUTE_PROVIDER_ADDRESS:-}" ] && ! grep -q '^EERFUL_0G_COMPUTE_PROVIDER_ADDRESS=' .env 2>/dev/null; then
    fail "  EERFUL_0G_COMPUTE_PROVIDER_ADDRESS not set in env or .env"
    PREFLIGHT_OK=false
else
    ok "  0G provider address              OK"
fi

# 5. Wallet free balance — soft warn, doesn't fail. Catches the case
#    where someone re-introduces the missing --skip-bridge-init flag
#    (would attempt a 1.1 0G add-ledger deposit per run) before the
#    on-chain failure shows up mid-recording. Threshold is 0.5 0G —
#    well above the per-call gas (~0.0003) and per-top-up cost
#    (~0.001 + gas) so a green light here means several hours of
#    recording without any wallet involvement.
WALLET_BAL_0G=$(python3 - <<'PY' 2>/dev/null
import json, urllib.request, sys
try:
    h = json.load(urllib.request.urlopen("http://127.0.0.1:7878/healthz", timeout=3))
    wallet, rpc = h["wallet"], h["rpc"]
    req = urllib.request.Request(
        rpc,
        data=json.dumps({"jsonrpc":"2.0","method":"eth_getBalance","params":[wallet,"latest"],"id":1}).encode(),
        headers={"Content-Type":"application/json"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=3))
    print(int(r["result"], 16) / 1e18)
except Exception as e:
    print("?", file=sys.stderr)
PY
)
if [ -z "$WALLET_BAL_0G" ] || [ "$WALLET_BAL_0G" = "?" ]; then
    note "  wallet balance                   (couldn't read — proceeding)"
elif awk -v b="$WALLET_BAL_0G" 'BEGIN{exit !(b < 0.5)}'; then
    fail "  wallet balance ${WALLET_BAL_0G} 0G   LOW"
    echo "      thin headroom for any tx-firing path (top-up, retrieve-fund,"
    echo "      or an accidental bridge_init re-fire). Faucet at:"
    echo "      https://faucet.0g.ai/  (your address: $(curl -sf http://127.0.0.1:7878/healthz | python3 -c 'import json,sys; print(json.load(sys.stdin)["wallet"])' 2>/dev/null))"
    echo "      or pull funds back via /admin/retrieve-fund."
else
    ok "  wallet balance ${WALLET_BAL_0G} 0G   OK"
fi

if ! $PREFLIGHT_OK; then
    echo
    fail "preflight failed — fix the above and re-run."
    exit 2
fi

if $CHECK_ONLY; then
    echo
    ok "preflight OK — exiting (--check-only)"
    exit 0
fi

# Reset receipt outputs so the demo starts from a known state.
rm -f examples/trading/receipts/proposal.json \
      examples/trading/receipts/implementation.json \
      examples/trading/receipts/proposal.md \
      examples/trading/receipts/implementation.py
mkdir -p examples/trading/receipts

POLICY=examples/trading/principal_policy.json
TIER=low_consequence
AGENT=examples/trading/axl/agent_multi.py

pause

# ─────────────────────────────────────────────────────────────────────
# CLEAN PATH
# ─────────────────────────────────────────────────────────────────────
scene "clean run — tool response"
note "the explorer (running on otto, talking AXL through gil) will read this."
note "innocuous market research. no mandate updates."
show_json examples/trading/tool_responses/clean.json
pause

scene "clean run — explorer drafts → refiner@louie optimizes (Optuna) → explorer renders"
note "AXL round trip over Yggdrasil (gil ↔ louie). Then 2 TEE receipts via 0G."
note "watch for: 'agent's working max_drawdown: 5%' (matches principal mandate)"
run uv run python "$AGENT" \
    --tool-response examples/trading/tool_responses/clean.json \
    --skip-bridge-init
pause

scene "clean run — gate proposal_grade  (expect PASS)"
run uv run eerful gate \
    --policy "$POLICY" --tier "$TIER" \
    --bundle proposal_grade \
    --receipt examples/trading/receipts/proposal.json
pause

scene "clean run — gate implementation_grade  (expect PASS)"
run uv run eerful gate \
    --policy "$POLICY" --tier "$TIER" \
    --bundle implementation_grade \
    --receipt examples/trading/receipts/implementation.json
echo
ok "${BOLD}both gates PASS — executor would run the strategy.${RESET}"
pause

# ─────────────────────────────────────────────────────────────────────
# POISONED PATH
# ─────────────────────────────────────────────────────────────────────
scene "poisoned run — tool response"
note "same agents. same code. same topology. only the tool response changed."
note "the 'mandate_updates' field is the attack — a forged regulatory update."
show_json examples/trading/tool_responses/poisoned.json
pause

scene "poisoned run — explorer drifts; refiner@louie faithfully optimizes the corrupted strategy"
note "watch for: 'agent's working max_drawdown: 30% (DRIFT — poisoned tool response)'"
run uv run python "$AGENT" \
    --tool-response examples/trading/tool_responses/poisoned.json \
    --skip-bridge-init
pause

scene "poisoned run — gate proposal_grade  (expect REFUSE_SCORE)"
note "the bundle's prompt pins the principal's TRUE mandate at 5%."
note "a 30% proposal scores mandate_compliance ≈ 0.3, overall ≈ 0.4 — below threshold 0.7."
set +e
uv run eerful gate \
    --policy "$POLICY" --tier "$TIER" \
    --bundle proposal_grade \
    --receipt examples/trading/receipts/proposal.json
PROPOSAL_RC=$?
set -e
case "$PROPOSAL_RC" in
    1) ok "${BOLD}proposal_grade REFUSED as expected (exit 1).${RESET}" ;;
    0) fail "UNEXPECTED PASS — proposal gate should have refused."; exit 1 ;;
    *) fail "wiring error (exit $PROPOSAL_RC) — check policy/receipt paths."; exit "$PROPOSAL_RC" ;;
esac
pause

scene "poisoned run — gate implementation_grade  (expect PASS — the two-gate point)"
note "the code faithfully implements the (poisoned) proposal — internally consistent."
note "a single-gate, implementation-only architecture would have let this through."
note "mandate drift is only catchable upstream where proposal_grade has a fixed reference"
note "(the principal's mandate hash) outside the agent's view."
run uv run eerful gate \
    --policy "$POLICY" --tier "$TIER" \
    --bundle implementation_grade \
    --receipt examples/trading/receipts/implementation.json
pause

# ─────────────────────────────────────────────────────────────────────
# COMPROMISED-AGENT PATH (forged receipt)
# ─────────────────────────────────────────────────────────────────────
scene "compromised-agent run — agent skips the TEE, signs its own receipt"
note "the prior arcs assumed an honest TEE call. now imagine the agent itself"
note "is compromised — it bypasses the enclave, generates its own key, and"
note "mints a receipt claiming high scores. all locally."
note "the forger borrows the prior clean run's attestation pointers so Steps"
note "2/4/5 (compose-hash) all pass; only Step 5b — pubkey ↔ report_data — catches it."
run uv run python examples/trading/axl/forge_attempt.py \
    --borrow-receipt examples/trading/receipts/proposal.json \
    --score 0.95 \
    --out examples/trading/receipts/forged.json
pause

scene "compromised-agent — gate (expect REFUSE_INVALID_RECEIPT, §7.1 Step 5b)"
note "Step 6 (signature → pubkey) passes — the forger signed their own"
note "response with their own key, the math is internally consistent."
note "Step 5b checks pubkey-derived address vs the attestation's report_data."
note "the forger's locally-generated key isn't bound to any enclave."
note "cryptographic refusal — different from the score-based refusal above."
# Capture combined stdout+stderr so we can assert the refusal CLASS,
# not just the exit code. `eerful gate` returns 1 for any REFUSE; if a
# future bug breaks the binding check but the receipt happens to
# refuse for some other reason (e.g. REFUSE_SCORE on a low-overall),
# the demo would silently still appear to "work." The grep pins the
# expected class.
set +e
FORGE_OUT="$(uv run eerful gate \
    --policy "$POLICY" --tier "$TIER" \
    --bundle proposal_grade \
    --receipt examples/trading/receipts/forged.json 2>&1)"
FORGE_RC=$?
set -e
printf '%s\n' "$FORGE_OUT"
case "$FORGE_RC" in
    1)
        if grep -q 'refuse_invalid_receipt' <<<"$FORGE_OUT"; then
            ok "${BOLD}forged receipt REFUSED as expected (REFUSE_INVALID_RECEIPT, Step 5b).${RESET}"
        else
            fail "UNEXPECTED REFUSE CLASS — expected REFUSE_INVALID_RECEIPT for forged receipt."
            exit 1
        fi
        ;;
    0) fail "UNEXPECTED PASS — forged receipt should have refused at Step 5b."; exit 1 ;;
    *) fail "wiring error (exit $FORGE_RC) — check policy/receipt paths."; exit "$FORGE_RC" ;;
esac

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
echo
echo "${BOLD}${GREEN}done.${RESET}"
echo "  clean path:        proposal PASS,   implementation PASS  → executor runs"
echo "  poisoned path:     proposal REFUSE, implementation PASS  → mandate-drift caught"
echo "  compromised-agent: forged receipt REFUSE                 → cryptographic refusal"
echo
note "the executor never sees a poisoned trade or a forged receipt. that's the rail."
note "(architecture-only N=4 high-consequence tier: \`eerful gate --tier high_consequence ...\` against any single receipt → REFUSE_INSUFFICIENT_RECEIPTS)"

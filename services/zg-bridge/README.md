# zg-bridge

Long-lived HTTP service that wraps `@0glabs/0g-serving-broker` and
`@0gfoundation/0g-ts-sdk` for the Python `eerful.zg.compute.ComputeClient`
and `eerful.zg.storage.BridgeStorageClient`. Both 0G SDKs are TypeScript;
the bridge keeps the Python side thin.

## Why a bridge

Both SDKs are TS-only. Direct ffi or shelling out per call were the
alternatives — the bridge wins on lifecycle (broker init takes a few
seconds; amortize once per process) and on isolation (the wallet key
lives in the bridge process, not the Python process).

## Security model

The bridge holds a hot wallet key in process and exposes admin/compute
endpoints with **no authentication**. The security model assumes the
only reachable client is the local Python adapter on the same host.

By default the bridge binds `127.0.0.1` and refuses to start on a
non-loopback host. To bind elsewhere you must set
`EERFUL_0G_BRIDGE_BIND_HOST_I_UNDERSTAND=true` AND front the service
with your own auth + network protection. **Don't expose this port to a
network without doing both.**

The `eerful` CLI (`src/eerful/cli.py`) refuses non-loopback bridge URLs
by default and requires `--allow-remote-bridge` to opt out. The Python
*library* clients (`BridgeStorageClient`, `ComputeClient`) accept any
URL — the loopback guard is only enforced at the CLI layer. Library
callers running their own pipelines are responsible for choosing safe
URLs themselves.

## Setup

```bash
cd services/zg-bridge
npm install
```

Create `eerful/.env` (one level above `services/zg-bridge/`) with at
minimum:

```bash
EERFUL_0G_PRIVATE_KEY=0x<your-galileo-testnet-key>
```

Optional overrides (all default sensibly):

| var                                 | default                                          | what it sets                              |
|-------------------------------------|--------------------------------------------------|-------------------------------------------|
| `EERFUL_0G_RPC`                     | `https://evmrpc-testnet.0g.ai`                   | Galileo EVM RPC endpoint                  |
| `EERFUL_0G_GALILEO_INDEXER`         | `https://indexer-storage-testnet-turbo.0g.ai`    | 0G Storage indexer                        |
| `EERFUL_0G_BRIDGE_PORT`             | `7878`                                           | port the bridge listens on                |
| `EERFUL_0G_BRIDGE_BIND_HOST`        | `127.0.0.1`                                      | bind host (loopback by default; see above)|
| `EERFUL_0G_BRIDGE_BIND_HOST_I_UNDERSTAND` | unset                                      | required `=true` to bind non-loopback     |

The wallet must be funded on Galileo testnet. Lockstep's measured
baseline is ~1.16 m0G per Storage upload; the demo's three-receipt
chain plus bundle/report uploads run well under 0.1 0G/day faucet
headroom.

## Running

```bash
# Foreground, with file-watch reload (dev)
npm run dev

# Foreground, no watcher (production-ish)
npm start

# Type-check only
npm run typecheck
```

On startup the bridge logs `zg-bridge listening on 127.0.0.1:7878`.
`/healthz` confirms the wallet is reachable and the broker initialized
successfully — use it to verify the bridge is ready before running
`eerful publish-evaluator` or producing a receipt.

## Endpoints

All endpoints return JSON unless noted. The Python clients in
`src/eerful/zg/{compute,storage}.py` are the canonical consumers — see
those modules for the request/response shapes and error contract.

### Health

- `GET /healthz` → `{status, wallet, chain_id, rpc, broker_initialized}`.
  Blocks until the broker finishes initializing on the first call after
  startup (subsequent calls hit the cached singleton). A 200 response
  implies `broker_initialized: true`; broker init failure surfaces as
  500 with `{status: "error", error: ...}`.

### Admin (one-time setup per wallet)

- `POST /admin/add-ledger` `{amount_0g: number}` — creates the broker
  ledger sub-account if absent (`addLedger`) or tops it up
  (`depositFund`). Broker's `MIN_LOCKED_BALANCE` for inference is 1 0G.
- `POST /admin/acknowledge` `{provider_address: string}` — acknowledges
  a TeeML provider so subsequent `/compute/inference` calls can bill
  against it. Per-provider, one-time.

### Compute

- `POST /compute/inference` `{provider_address, messages, temperature?, max_tokens?}`
  → `{chat_id, response_content, model_served, provider_endpoint, usage}`.
  Drives the broker's signed-billing-headers flow and forwards the
  upstream OpenAI-shape `usage` (`{input_tokens, output_tokens}`) so
  jig's `BudgetTracker` can see EER calls. Strict validation:
  token counts must be finite non-negative integers.
- `GET /compute/signature/:chat_id?provider=<addr>`
  → `{signature_hex, message_text}`. Fetches the per-call signature the
  TEE produced over the response body. The Python side recovers the
  enclave pubkey from `(message_text, signature_hex)` for spec §6.4
  receipt construction.
- `GET /compute/attestation/:provider_address`
  → raw bytes + `X-Report-Hash: 0x<sha256>` header,
  `Content-Type: application/octet-stream`. Provider's current
  attestation report. Fetched fresh on every request via the broker
  SDK's `getQuote(provider_address)`; not cached locally. Producers
  generating a chain of receipts back-to-back will hit the provider
  per round; consider caching at the call site if that matters.

### Storage

- `POST /storage/upload-blob` (body: raw bytes, `Content-Type:
  application/octet-stream`) → `{content_hash, storage_uri, root_hash,
  tx_hash, tx_seq, size_bytes}`. `content_hash` is the sha256 (receipt
  field `evaluator_id` / `attestation_report_hash`); `root_hash` is the
  0G storage Merkle root (receipt field `evaluator_storage_root` /
  `attestation_storage_root`). Both flow into the receipt; the verifier
  passes them to `/download-blob`. `storage_uri` is the convenience
  `zg://<root_hash>` form. Idempotent on `content_hash`. Coalesces
  concurrent uploads of identical bytes so two callers don't both pay
  the upload fee.
- `GET /storage/download-blob?content_hash=0x...&root_hash=0x...` → raw
  bytes. Both query params are required: `root_hash` is the indexer
  lookup key, `content_hash` is the integrity check (sha256 of the
  returned bytes must equal it). Cross-instance safe — any bridge can
  serve any blob the indexer has, no producer-side state required.
  Returns 422 if the indexer's bytes don't re-hash to the requested
  `content_hash`; 502 if the indexer can't resolve `root_hash`.

## Error contract

`BridgeStorageClient` (the storage adapter) depends on this status →
exception mapping; change it and the storage adapter breaks.

- 200: success.
- 400: malformed request (programming bug on the caller's side).
  `BridgeStorageClient` raises `RuntimeError`.
- 422: byzantine evidence — bytes returned don't re-hash to the
  requested content hash. `BridgeStorageClient` raises
  `TrustViolation`. Never retry.
- 5xx: SDK / indexer / RPC failure (including 502 when `root_hash`
  doesn't resolve). `BridgeStorageClient` raises `StorageError`.
  Transient.

`ComputeClient` is simpler: any non-2xx response from the compute /
admin endpoints raises `ComputeError(f"{op} failed ({status}): ...")`.
There is no per-status mapping on the compute side; callers handle
or propagate `ComputeError` uniformly.

Error responses are JSON `{error: string, detail?: any}` regardless of
which client consumes them, so the message surfaces rather than the
raw 500.

## Limitations

- Single-tenant. The wallet is shared across all consumers of this
  bridge instance — multi-tenancy needs separate processes.
- `uploadIndex` (sha256 → 0G rootHash) is in-memory upload-side
  dedup only. Resets on restart, which means a restarted bridge will
  re-pay the SDK upload call the next time the same bytes are
  uploaded (the indexer short-circuits on the already-stored
  rootHash, but the bridge re-bills the call). Downloads are
  unaffected: receipts carry both `content_hash` and `root_hash`, so
  any bridge can fetch any receipt's bytes without producer-side
  state.
- No rate limiting. The loopback-only default is what protects against
  abuse; remote operators are responsible for their own budget caps.

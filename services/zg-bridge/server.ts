/**
 * eerful zg-bridge — HTTP wrapper around @0glabs/0g-serving-broker.
 *
 * Why: the broker SDK is TypeScript-only. eerful is a Python package. This
 * service runs locally (default 127.0.0.1:7878), holds the wallet, and
 * exposes the inference flow as plain HTTP so the Python ComputeClient can
 * stay thin.
 *
 * Day 2 surface: TeeML compute only (no storage yet — Day 3 adds upload/
 * download blob endpoints).
 *
 * Security: the wallet key is in-process. Bind 127.0.0.1 unless the
 * operator explicitly overrides EERFUL_0G_BRIDGE_BIND_HOST AND fronts the
 * service with their own auth. Never expose this port to a network.
 */

import * as path from 'path';
import * as crypto from 'crypto';
import * as dotenv from 'dotenv';
import express from 'express';
import type { Request, Response } from 'express';
import { ethers } from 'ethers';
import {
  createZGComputeNetworkBroker,
  ZGComputeNetworkBroker,
} from '@0glabs/0g-serving-broker';
import { Indexer, MemData } from '@0gfoundation/0g-ts-sdk';

dotenv.config({ path: path.resolve(__dirname, '../../.env') });

const PRIVATE_KEY = process.env.EERFUL_0G_PRIVATE_KEY;
if (!PRIVATE_KEY || PRIVATE_KEY === '0x_replace_with_real_key') {
  console.error(
    'zg-bridge: EERFUL_0G_PRIVATE_KEY is not set. ' +
      'Mirror lockstep/.env into eerful/.env with EERFUL_ prefixes.',
  );
  process.exit(1);
}

const RPC_URL = process.env.EERFUL_0G_RPC ?? 'https://evmrpc-testnet.0g.ai';
const INDEXER_URL =
  process.env.EERFUL_0G_GALILEO_INDEXER ??
  'https://indexer-storage-testnet-turbo.0g.ai';
const PORT = Number(process.env.EERFUL_0G_BRIDGE_PORT ?? '7878');
const BIND_HOST = process.env.EERFUL_0G_BRIDGE_BIND_HOST ?? '127.0.0.1';

// The bridge holds a hot wallet key in process and exposes admin/compute
// endpoints with no authentication (the security model assumes the only
// reachable client is the local Python adapter). Refuse to start on a
// non-loopback bind unless the operator opts in explicitly. Without this
// guard, a fat-fingered EERFUL_0G_BRIDGE_BIND_HOST=0.0.0.0 hands the
// wallet to anyone on the network.
function normalizeBindHost(host: string): string {
  const trimmed = host.trim();
  if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function isLoopbackBindHost(host: string): boolean {
  const normalized = normalizeBindHost(host).toLowerCase();
  return normalized === '127.0.0.1' || normalized === '::1' || normalized === 'localhost';
}

const ALLOW_NON_LOOPBACK_BIND =
  (process.env.EERFUL_0G_BRIDGE_BIND_HOST_I_UNDERSTAND ?? '').toLowerCase() === 'true';

if (!isLoopbackBindHost(BIND_HOST) && !ALLOW_NON_LOOPBACK_BIND) {
  console.error(
    `zg-bridge: refusing to bind to non-loopback host "${BIND_HOST}". ` +
      'This service holds a hot private key and exposes unauthenticated endpoints. ' +
      'Use EERFUL_0G_BRIDGE_BIND_HOST=127.0.0.1 (or ::1), or set ' +
      'EERFUL_0G_BRIDGE_BIND_HOST_I_UNDERSTAND=true if you accept the risk and ' +
      'are providing your own network protections.',
  );
  process.exit(1);
}

const provider = new ethers.JsonRpcProvider(RPC_URL);
const wallet = new ethers.Wallet(PRIVATE_KEY, provider);
const indexer = new Indexer(INDEXER_URL);

// sha256(content) -> 0G storage rootHash + tx metadata. The receipt format
// is content-addressed by sha256 (evaluator_id, attestation_report_hash);
// 0G's indexer keys by Merkle root. This index is the bridge between the
// two. Resets on restart — bytes persist on 0G but the mapping doesn't, so
// a publisher who restarts the bridge must re-upload to repopulate
// (idempotent: getOrUpload short-circuits on cache hit, but the cache is
// what's gone, so the upload will go through and rebuild the entry).
type UploadIndexEntry = {
  zgRoot: string;
  txHash: string;
  txSeq: number;
  size: number;
};
const uploadIndex = new Map<string, UploadIndexEntry>();

// In-flight upload coalescing. Two requests carrying identical bytes
// that arrive concurrently both miss `uploadIndex`; without coalescing
// they would each call `indexer.upload` and pay separately. Same
// pattern as `getBroker`'s `brokerInit`. Cleared in finally().
type UploadResult = [UploadIndexEntry, null] | [null, Error];
const inflightUploads = new Map<string, Promise<UploadResult>>();

let broker: ZGComputeNetworkBroker | null = null;
let brokerInit: Promise<ZGComputeNetworkBroker> | null = null;

async function getBroker(): Promise<ZGComputeNetworkBroker> {
  if (broker !== null) return broker;
  if (brokerInit !== null) return brokerInit;
  // Cache the in-flight init promise so concurrent first-callers share it
  // instead of each invoking createZGComputeNetworkBroker (which races on
  // ledger / signer state and last-write-wins on `broker`).
  brokerInit = createZGComputeNetworkBroker(wallet)
    .then((b) => {
      broker = b;
      return b;
    })
    .finally(() => {
      brokerInit = null;
    });
  return brokerInit;
}

function sha256Hex(bytes: Buffer | Uint8Array): string {
  return '0x' + crypto.createHash('sha256').update(bytes).digest('hex');
}

const app = express();
app.use(express.json({ limit: '4mb' }));

// ---------------- /healthz ----------------

app.get('/healthz', async (_req: Request, res: Response) => {
  try {
    const b = await getBroker();
    const network = await provider.getNetwork();
    res.json({
      status: 'ok',
      wallet: wallet.address,
      chain_id: Number(network.chainId),
      rpc: RPC_URL,
      broker_initialized: b !== null,
    });
  } catch (err) {
    res.status(500).json({
      status: 'error',
      error: err instanceof Error ? err.message : String(err),
    });
  }
});

// ---------------- /admin/add-ledger ----------------
//
// Creates the ledger sub-account if absent (addLedger), or tops it up
// (depositFund) if present. Required ONCE before any provider can be
// acknowledged or paid. Body: { amount_0g: number } in 0G units, e.g. 0.5.
// The broker's MIN_LOCKED_BALANCE for inference is 1 0G; deposits below
// that won't satisfy provider lock requirements at request time.

app.post('/admin/add-ledger', async (req: Request, res: Response) => {
  const { amount_0g } = req.body ?? {};
  if (typeof amount_0g !== 'number' || amount_0g <= 0) {
    res.status(400).json({ error: "missing or invalid 'amount_0g'" });
    return;
  }
  try {
    const b = await getBroker();
    let existed = true;
    try {
      await b.ledger.getLedger();
    } catch (probe) {
      existed = false;
      console.warn(
        `add-ledger: getLedger() probe failed, assuming ledger missing — ${
          probe instanceof Error ? probe.message : String(probe)
        }`,
      );
    }
    // The SDK doesn't surface a clean "not found" error code for ledgers, so
    // we treat any getLedger() failure as "missing." Log the actual error so
    // network/RPC issues aren't silently misclassified as a missing ledger.
    if (existed) {
      await b.ledger.depositFund(amount_0g);
    } else {
      await b.ledger.addLedger(amount_0g);
    }
    const ledger = await b.ledger.getLedger();
    res.json({
      created: !existed,
      total_balance_neuron: ledger.totalBalance.toString(),
      available_balance_neuron: ledger.availableBalance.toString(),
      total_balance_0g: Number(ledger.totalBalance) / 1e18,
    });
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : String(err),
    });
  }
});

// ---------------- /admin/acknowledge ----------------
//
// One-time per (wallet, provider). Idempotent — the SDK no-ops if the
// signer is already acknowledged. Returns the on-chain teeSignerAddress
// so callers can confirm what they're trusting.

app.post('/admin/acknowledge', async (req: Request, res: Response) => {
  const { provider_address } = req.body ?? {};
  if (typeof provider_address !== 'string') {
    res.status(400).json({ error: "missing 'provider_address'" });
    return;
  }
  try {
    const b = await getBroker();
    const preStatus = await b.inference.checkProviderSignerStatus(provider_address);
    let finalStatus = preStatus;
    if (!preStatus.isAcknowledged) {
      await b.inference.acknowledgeProviderSigner(provider_address);
      // Re-check after the on-chain ack so callers see the current
      // teeSignerAddress, not whatever was bound pre-ack.
      finalStatus = await b.inference.checkProviderSignerStatus(provider_address);
    }
    res.json({
      provider_address,
      tee_signer_address: finalStatus.teeSignerAddress,
      already_acknowledged: preStatus.isAcknowledged,
    });
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : String(err),
    });
  }
});

// ---------------- /compute/inference ----------------
//
// Body: { provider_address, messages: [{role, content}], temperature?, max_tokens? }
// Returns: { chat_id, response_content, model_served, provider_endpoint }
// (signing_address is recovered Python-side from the signature endpoint's
//  response, not returned here.)
//
// Flow:
//   1. getServiceMetadata → endpoint + on-chain model name
//   2. getRequestHeaders(content) → billing headers (signed by wallet)
//   3. POST OpenAI-compatible chat completion to {endpoint}/chat/completions
//   4. Extract chat_id from ZG-Res-Key header (fall back to completion.id)
//   5. processResponse → caches fee + verifies signature

app.post('/compute/inference', async (req: Request, res: Response) => {
  const { provider_address, messages, temperature, max_tokens } = req.body ?? {};
  if (typeof provider_address !== 'string' || !Array.isArray(messages)) {
    res.status(400).json({ error: "missing 'provider_address' or 'messages'" });
    return;
  }
  // Validate each message has string role + content. Without this, malformed
  // input either throws a 500 inside the broker SDK or, worse, signs/bills
  // `undefined` content silently.
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (
      m === null ||
      typeof m !== 'object' ||
      typeof (m as { role?: unknown }).role !== 'string' ||
      typeof (m as { content?: unknown }).content !== 'string'
    ) {
      res.status(400).json({
        error: `messages[${i}] must be { role: string, content: string }`,
      });
      return;
    }
  }
  try {
    const b = await getBroker();
    const meta = await b.inference.getServiceMetadata(provider_address);

    // Spec §6.5 inference_params: only the user message bytes are billed/signed.
    // Concatenate user contents so the broker's content-fee accounting matches.
    const userContent = messages
      .filter((m: { role: string }) => m.role === 'user')
      .map((m: { content: string }) => m.content)
      .join('\n');

    const headers = await b.inference.getRequestHeaders(provider_address, userContent);

    const body = {
      messages,
      model: meta.model,
      ...(temperature !== undefined ? { temperature } : {}),
      ...(max_tokens !== undefined ? { max_tokens } : {}),
    };

    const upstream = await fetch(`${meta.endpoint}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(body),
    });

    if (!upstream.ok) {
      const text = await upstream.text();
      res.status(upstream.status).json({
        error: 'upstream provider rejected request',
        status: upstream.status,
        body: text,
      });
      return;
    }

    const chatId = upstream.headers.get('ZG-Res-Key') ?? '';
    const completion = (await upstream.json()) as {
      id: string;
      choices: Array<{ message: { content: string } }>;
      usage?: unknown;
    };
    const effectiveChatId = chatId || completion.id;
    const responseContent = completion.choices?.[0]?.message?.content ?? '';

    // processResponse caches usage-fee + verifies the upstream signature.
    // Pass the usage block as the content arg per SDK contract.
    const usageJson = JSON.stringify(completion.usage ?? {});
    await b.inference.processResponse(provider_address, effectiveChatId, usageJson);

    res.json({
      chat_id: effectiveChatId,
      response_content: responseContent,
      model_served: meta.model,
      provider_endpoint: meta.endpoint,
    });
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : String(err),
    });
  }
});

// ---------------- /compute/signature/:chat_id ----------------
//
// Returns: { signature_hex, message_text } — the provider's per-chat
// signature and the canonical text it was signed over. Address binding
// (recovered pubkey ↔ on-chain teeSignerAddress) is verified Python-side
// in ComputeClient.infer_full, not at the bridge boundary.

app.get('/compute/signature/:chat_id', async (req: Request, res: Response) => {
  const chatId = req.params.chat_id;
  const providerAddress = req.query.provider;
  if (typeof providerAddress !== 'string' || !chatId) {
    res.status(400).json({ error: "missing 'provider' query param or chat_id" });
    return;
  }
  try {
    const b = await getBroker();
    const link = await b.inference.getChatSignatureDownloadLink(providerAddress, chatId);
    const upstream = await fetch(link);
    if (!upstream.ok) {
      res.status(upstream.status).json({
        error: 'upstream provider rejected signature fetch',
        status: upstream.status,
      });
      return;
    }
    // The provider's signature endpoint returns JSON: { signature, text }
    // (text is the message that was signed; canonicalized by the provider).
    const payload = (await upstream.json()) as { signature: string; text: string };
    res.json({
      signature_hex: payload.signature,
      message_text: payload.text,
    });
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : String(err),
    });
  }
});

// ---------------- /compute/attestation/:provider_address ----------------
//
// Returns the provider's current attestation report (raw bytes) plus the
// content hash. The Python client uses the hash for `attestation_report_hash`
// in the receipt (Step 4 will fetch and re-hash; here we precompute so the
// bridge boundary already commits to specific bytes).

app.get('/compute/attestation/:provider_address', async (req: Request, res: Response) => {
  const providerAddress = req.params.provider_address;
  if (!providerAddress) {
    res.status(400).json({ error: "missing 'provider_address'" });
    return;
  }
  try {
    const b = await getBroker();
    // getQuote() does not hit disk; downloadQuoteReport() does. We want bytes.
    const result = await b.inference.requestProcessor.getQuote(providerAddress);
    // Today the SDK's getQuote returns rawReport as a UTF-8 string (the JSON
    // body of /v1/quote). Guard explicitly so a future SDK change to bytes /
    // base64 / Uint8Array fails loudly here instead of silently corrupting
    // the hash and producing receipts that can't be verified.
    if (typeof result.rawReport !== 'string') {
      throw new Error(
        `getQuote returned non-string rawReport (${typeof result.rawReport}); ` +
          `update bridge to handle the new SDK shape before producing receipts`,
      );
    }
    const reportBytes = Buffer.from(result.rawReport, 'utf8');
    const reportHash = sha256Hex(reportBytes);
    res.set('Content-Type', 'application/octet-stream');
    res.set('X-Report-Hash', reportHash);
    res.send(reportBytes);
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : String(err),
    });
  }
});

// ---------------- storage helpers ----------------

// Idempotent upload. If `contentHash` is already in `uploadIndex`,
// returns the cached entry without re-paying. Otherwise uploads via the
// SDK and caches the result. Same shape as lockstep's storage-ts.
async function getOrUpload(
  bytes: Buffer,
  contentHash: string,
): Promise<UploadResult> {
  const cached = uploadIndex.get(contentHash);
  if (cached) {
    return [cached, null];
  }
  const inflight = inflightUploads.get(contentHash);
  if (inflight) {
    return inflight;
  }
  const p: Promise<UploadResult> = (async (): Promise<UploadResult> => {
    const [tx, err] = await indexer.upload(new MemData(bytes), RPC_URL, wallet);
    if (err !== null) {
      return [null, err];
    }
    // Defensive: SDK contract says err === null implies a tx with
    // rootHash, but null/undefined or a rootHash-less object would
    // crash the property access below. Fail loudly instead.
    if (tx == null || typeof tx !== 'object' || !('rootHash' in tx)) {
      return [null, new Error('unknown SDK upload failure: missing rootHash on tx')];
    }
    const entry: UploadIndexEntry = {
      zgRoot: tx.rootHash,
      txHash: tx.txHash,
      txSeq: tx.txSeq,
      size: bytes.length,
    };
    uploadIndex.set(contentHash, entry);
    return [entry, null];
  })().finally(() => {
    inflightUploads.delete(contentHash);
  });
  inflightUploads.set(contentHash, p);
  return p;
}

const CONTENT_HASH_RE = /^0x[0-9a-f]{64}$/;

// ---------------- /storage/upload-blob ----------------
//
// Body: raw bytes (Content-Type: application/octet-stream).
// Returns: { content_hash, storage_uri, root_hash, tx_hash, tx_seq, size_bytes }
// content_hash is sha256 of the body and is what the Python adapter
// passes back to /storage/download-blob.
//
// Idempotent: a retry on the same bytes hits the cache and does not re-pay.

app.post(
  '/storage/upload-blob',
  // Match only application/octet-stream so the global express.json
  // middleware can't double-parse a misrouted JSON request before the
  // raw body parser sees it. A request with the wrong Content-Type
  // falls through to an empty req.body and 400s on the explicit check.
  express.raw({ type: 'application/octet-stream', limit: '8mb' }),
  async (req: Request, res: Response) => {
    try {
      const bytes = req.body as Buffer;
      if (!Buffer.isBuffer(bytes) || bytes.length === 0) {
        res.status(400).json({
          error: 'empty_body',
          detail: 'POST body required (Content-Type: application/octet-stream)',
        });
        return;
      }
      const contentHash = sha256Hex(bytes);
      const [entry, err] = await getOrUpload(bytes, contentHash);
      if (err !== null) {
        res.status(502).json({
          error: 'upload_failed',
          detail: err.message ?? String(err),
        });
        return;
      }
      res.json({
        content_hash: contentHash,
        storage_uri: `zg://${entry.zgRoot}`,
        root_hash: entry.zgRoot,
        tx_hash: entry.txHash,
        tx_seq: entry.txSeq,
        size_bytes: entry.size,
      });
    } catch (e: unknown) {
      res.status(500).json({
        error: 'internal',
        detail: e instanceof Error ? e.message : String(e),
      });
    }
  },
);

// ---------------- /storage/download-blob ----------------
//
// Query: ?content_hash=0x<sha256>
// Returns: raw bytes + X-Content-Hash header.
//
// 404 when the bridge has no upload record for the content hash (the
// bytes may exist on 0G under their rootHash, but this process did not
// upload them — see uploadIndex docstring). 422 if the downloaded bytes
// don't re-hash to the requested content_hash (defense-in-depth against
// indexer/SDK swaps).

app.get('/storage/download-blob', async (req: Request, res: Response) => {
  const contentHash = String(req.query.content_hash ?? '').toLowerCase();
  if (!CONTENT_HASH_RE.test(contentHash)) {
    res.status(400).json({
      error: 'invalid_content_hash',
      detail: 'content_hash must be 0x-prefixed lowercase 64-hex',
    });
    return;
  }
  const entry = uploadIndex.get(contentHash);
  if (!entry) {
    res.status(404).json({
      error: 'not_in_index',
      detail:
        'content_hash has no upload record on this bridge ' +
        '(different uploader, or bridge restarted since upload)',
    });
    return;
  }
  try {
    const [blob, err] = await indexer.downloadToBlob(entry.zgRoot);
    if (err !== null) {
      res.status(502).json({
        error: 'download_failed',
        detail: err.message ?? String(err),
      });
      return;
    }
    const bytes = Buffer.from(await blob.arrayBuffer());
    const actual = sha256Hex(bytes);
    if (actual !== contentHash) {
      res.status(422).json({
        error: 'content_hash_mismatch',
        detail: `downloaded bytes sha256 ${actual} != requested ${contentHash}`,
      });
      return;
    }
    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('X-Content-Hash', contentHash);
    res.setHeader('X-Root-Hash', entry.zgRoot);
    res.send(bytes);
  } catch (e: unknown) {
    res.status(500).json({
      error: 'internal',
      detail: e instanceof Error ? e.message : String(e),
    });
  }
});

// ---------------- start ----------------

app.listen(PORT, BIND_HOST, () => {
  console.log(
    `zg-bridge listening on ${BIND_HOST}:${PORT} ` +
      `(wallet=${wallet.address}, rpc=${RPC_URL})`,
  );
});

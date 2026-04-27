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
const PORT = Number(process.env.EERFUL_0G_BRIDGE_PORT ?? '7878');
const BIND_HOST = process.env.EERFUL_0G_BRIDGE_BIND_HOST ?? '127.0.0.1';

const provider = new ethers.JsonRpcProvider(RPC_URL);
const wallet = new ethers.Wallet(PRIVATE_KEY, provider);

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
    const status = await b.inference.checkProviderSignerStatus(provider_address);
    if (!status.isAcknowledged) {
      await b.inference.acknowledgeProviderSigner(provider_address);
    }
    res.json({
      provider_address,
      tee_signer_address: status.teeSignerAddress,
      already_acknowledged: status.isAcknowledged,
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

// ---------------- start ----------------

app.listen(PORT, BIND_HOST, () => {
  console.log(
    `zg-bridge listening on ${BIND_HOST}:${PORT} ` +
      `(wallet=${wallet.address}, rpc=${RPC_URL})`,
  );
});

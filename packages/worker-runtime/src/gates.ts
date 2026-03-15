// ─── Primitive Gate Evaluation ────────────────────────────────────────────────
//
// The runner evaluates gates locally using a registry of primitive op handlers.
// The cloud sends WHAT to check (op + params). The runner decides HOW.
// The cloud receives an outcome — never the underlying data.

import { logger } from "./logger.js";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface GateRequest {
  /** Unique gate evaluation ID (assigned by cloud) */
  gateId: string;
  /** Entity ID this gate is being evaluated against */
  entityId: string;
  /** Primitive op identifier, e.g. "vcs.ci_status" */
  op: string;
  /** Handlebars-rendered params for the op */
  params: Record<string, unknown>;
  /** Timeout in milliseconds. Runner enforces this. */
  timeoutMs?: number;
}

export interface GateResult {
  gateId: string;
  entityId: string;
  op: string;
  outcome: string;
  message: string;
  /** Duration of gate evaluation in milliseconds */
  durationMs: number;
}

/**
 * A primitive op handler. Receives the op name, rendered params, and entity context.
 * Returns an outcome string (matched against the gate's outcomes map by the cloud)
 * and an optional human-readable message.
 */
export type PrimitiveHandler = (
  op: string,
  params: Record<string, unknown>,
  context: { entityId: string; signal?: AbortSignal },
) => Promise<{ outcome: string; message?: string }>;

// ─── Registry ────────────────────────────────────────────────────────────────

const handlers = new Map<string, PrimitiveHandler>();

/**
 * Register a primitive op handler.
 * Exact match on op name (e.g. "vcs.ci_status").
 */
export function registerHandler(op: string, handler: PrimitiveHandler): void {
  if (handlers.has(op)) {
    logger.warn(`[gates] overwriting handler for op "${op}"`);
  }
  handlers.set(op, handler);
  logger.info(`[gates] registered handler for op "${op}"`);
}

/**
 * Register multiple handlers at once.
 */
export function registerHandlers(entries: Record<string, PrimitiveHandler>): void {
  for (const [op, handler] of Object.entries(entries)) {
    registerHandler(op, handler);
  }
}

/**
 * Check if a handler is registered for the given op.
 */
export function hasHandler(op: string): boolean {
  return handlers.has(op);
}

/**
 * List all registered op names.
 */
export function listHandlers(): string[] {
  return [...handlers.keys()].sort();
}

/**
 * Clear all registered handlers. Primarily for testing.
 */
export function clearHandlers(): void {
  handlers.clear();
}

// ─── Evaluation ──────────────────────────────────────────────────────────────

const DEFAULT_TIMEOUT_MS = 300_000; // 5 minutes

/**
 * Evaluate a single gate request against the handler registry.
 */
export async function evaluateGate(request: GateRequest): Promise<GateResult> {
  const { gateId, entityId, op, params, timeoutMs } = request;
  const timeout = timeoutMs != null && timeoutMs > 0 ? timeoutMs : DEFAULT_TIMEOUT_MS;
  const start = Date.now();

  logger.info(`[gates] evaluating gate`, { gateId, entityId, op, paramKeys: Object.keys(params) });

  const handler = handlers.get(op);
  if (!handler) {
    const durationMs = Date.now() - start;
    logger.warn(`[gates] no handler registered for op "${op}"`, { gateId, entityId });
    return {
      gateId,
      entityId,
      op,
      outcome: "error",
      message: `No handler registered for primitive op "${op}"`,
      durationMs,
    };
  }

  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeoutPromise = new Promise<{ outcome: string; message?: string }>((resolve) => {
    timer = setTimeout(() => {
      controller.abort();
      resolve({ outcome: "timeout", message: `Gate timed out after ${timeout}ms` });
    }, timeout);
  });

  try {
    const result = await Promise.race([handler(op, params, { entityId, signal: controller.signal }), timeoutPromise]);

    const durationMs = Date.now() - start;
    logger.info(`[gates] gate evaluated`, {
      gateId,
      entityId,
      op,
      outcome: result.outcome,
      durationMs,
    });

    return {
      gateId,
      entityId,
      op,
      outcome: result.outcome,
      message: result.message ?? "",
      durationMs,
    };
  } catch (err) {
    const durationMs = Date.now() - start;
    const message = err instanceof Error ? err.message : String(err);
    logger.error(`[gates] handler threw`, { gateId, entityId, op, error: message });

    return {
      gateId,
      entityId,
      op,
      outcome: "error",
      message,
      durationMs,
    };
  } finally {
    clearTimeout(timer);
  }
}

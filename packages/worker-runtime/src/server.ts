import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile } from "node:fs/promises";
import type { IncomingMessage, RequestListener, ServerResponse } from "node:http";
import { join } from "node:path";
import { promisify } from "node:util";
import { createOpencode } from "@opencode-ai/sdk";
import { evaluateGate, type GateRequest, listHandlers } from "./gates.js";
import { logger } from "./logger.js";
import { parseSignal } from "./parse-signal.js";
import type { DispatchRequest, HolyshipperEvent } from "./types.js";

const execFileAsync = promisify(execFile);

/**
 * Model map: tier → OpenRouter model IDs.
 * These go through our gateway, which proxies to OpenRouter.
 * Note: providerID/modelID are sent as flat body fields (not nested in model object)
 * because the OpenCode server API expects them at the body root level.
 */
const MODEL_MAP: Record<DispatchRequest["modelTier"], { providerID: string; modelID: string }> = {
  opus: { providerID: "holyship", modelID: "anthropic/claude-opus-4-6" },
  sonnet: { providerID: "holyship", modelID: "anthropic/claude-sonnet-4-6" },
  haiku: { providerID: "holyship", modelID: "anthropic/claude-haiku-4-5" },
  deepseek: { providerID: "holyship", modelID: "deepseek/deepseek-v3.2" },
  test: { providerID: "holyship", modelID: "qwen/qwen3-coder" },
};

/**
 * Lazily-initialized OpenCode client.
 * The server starts once and the client is reused across dispatches.
 * If the server process dies, we detect it on next call and re-init.
 */
let _opencodeClient: Awaited<ReturnType<typeof createOpencode>> | null = null;

async function getOpencode() {
  // Health check: if we have a client, verify the server is still alive
  if (_opencodeClient) {
    try {
      await _opencodeClient.client.path.get();
    } catch (err) {
      logger.warn("[opencode] server health check failed, reinitializing", {
        error: err instanceof Error ? err.message : String(err),
      });
      try {
        _opencodeClient.server.close();
      } catch {
        /* already dead */
      }
      _opencodeClient = null;
    }
  }

  if (_opencodeClient) return _opencodeClient;

  const gatewayUrl = process.env.HOLYSHIP_GATEWAY_URL ?? "http://localhost:3001/v1";

  logger.info("[opencode] initializing server", {
    gatewayUrl,
    hasGatewayKey: !!process.env.HOLYSHIP_GATEWAY_KEY,
  });

  // Write opencode.json so the Go server registers our custom provider.
  // The Go server reads config from disk, not from the SDK config param.
  // Models MUST be declared explicitly or the server returns ProviderModelNotFoundError.
  const configPath = join(process.cwd(), "opencode.json");
  const { writeFileSync } = await import("node:fs");
  writeFileSync(
    configPath,
    JSON.stringify({
      $schema: "https://opencode.ai/config.json",
      provider: {
        holyship: {
          npm: "@ai-sdk/openai-compatible",
          name: "Holy Ship Gateway",
          env: ["HOLYSHIP_GATEWAY_KEY"],
          options: { baseURL: gatewayUrl },
          models: {
            "anthropic/claude-opus-4-6": { name: "Claude Opus" },
            "anthropic/claude-sonnet-4-6": { name: "Claude Sonnet" },
            "anthropic/claude-haiku-4-5": { name: "Claude Haiku" },
            "deepseek/deepseek-v3.2": { name: "DeepSeek V3.2" },
            "qwen/qwen3-coder": { name: "Qwen3 Coder" },
            "openai/gpt-4o-mini": { name: "GPT-4o Mini" },
            "openai/gpt-4o": { name: "GPT-4o" },
            "nousresearch/hermes-3-llama-3.1-405b:free": { name: "Hermes 405B (free)" },
          },
        },
      },
    }),
  );
  logger.info("[opencode] wrote opencode.json", { configPath, gatewayUrl });

  _opencodeClient = await createOpencode({ timeout: 15000 });

  logger.info("[opencode] server started", { url: _opencodeClient.server.url });
  return _opencodeClient;
}

/**
 * Auto-accept a permission request.
 * OpenCode pauses agent execution until permissions are responded to.
 * In headless mode (holyshipper), we always accept.
 */
async function autoAcceptPermission(
  client: Awaited<ReturnType<typeof createOpencode>>["client"],
  sessionId: string,
  permissionId: string,
): Promise<void> {
  try {
    await client.postSessionIdPermissionsPermissionId({
      path: { id: sessionId, permissionID: permissionId },
      body: { response: "always" },
    });
    logger.info("[dispatch:permission] auto-accepted", { sessionId, permissionId });
  } catch (err) {
    logger.error("[dispatch:permission] auto-accept failed", {
      sessionId,
      permissionId,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

const MAX_BODY_SIZE = 1024 * 1024;
const WORKSPACE = "/workspace";
const GH_TOKEN_PATH = "/run/secrets/gh-token";

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let totalSize = 0;
    req.on("data", (chunk: Buffer) => {
      totalSize += chunk.length;
      if (totalSize > MAX_BODY_SIZE) {
        req.resume();
        reject(new Error("Request body too large"));
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

function sendSSE(res: ServerResponse, event: HolyshipperEvent): void {
  res.write(`data: ${JSON.stringify(event)}\n\n`);
}

/** Read GH token from /run/secrets, env, or credentials injection. */
async function resolveGhToken(): Promise<string | null> {
  // 1. Env var (set by credentials injection)
  if (process.env.GH_TOKEN) return process.env.GH_TOKEN;
  if (process.env.GITHUB_TOKEN) return process.env.GITHUB_TOKEN;
  // 2. Secrets file (mounted by HolyshipperDispatcher; overridable in tests via GH_TOKEN_PATH_OVERRIDE)
  const tokenPath = process.env.GH_TOKEN_PATH_OVERRIDE ?? GH_TOKEN_PATH;
  if (existsSync(tokenPath)) {
    return (await readFile(tokenPath, "utf-8")).trim();
  }
  return null;
}

async function handleDispatch(req: IncomingMessage, res: ServerResponse): Promise<void> {
  let body: string;
  try {
    body = await readBody(req);
  } catch (err) {
    logger.warn(`[dispatch] body read failed`, { error: err instanceof Error ? err.message : String(err) });
    res.writeHead(400).end("Bad request");
    res.on("finish", () => req.destroy());
    return;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    logger.warn(`[dispatch] invalid JSON body`);
    res.writeHead(400).end("Invalid JSON");
    return;
  }

  const data = parsed as Record<string, unknown>;
  if (typeof data.prompt !== "string" || !data.prompt) {
    logger.warn(`[dispatch] missing prompt field`);
    res.writeHead(400).end("Missing prompt");
    return;
  }

  const modelTier = (data.modelTier as DispatchRequest["modelTier"]) ?? "sonnet";
  const model = MODEL_MAP[modelTier];
  const sessionId =
    data.newSession === true ? undefined : typeof data.sessionId === "string" ? data.sessionId : undefined;

  const promptPreview = (data.prompt as string).slice(0, 200);
  logger.info(`[dispatch] received`, {
    modelTier,
    model,
    sessionId: sessionId ?? "(new)",
    promptLength: (data.prompt as string).length,
    promptPreview,
  });

  const allText: string[] = [];
  const startTime = Date.now();
  let toolUseCount = 0;
  let textBlockCount = 0;
  let resolvedSessionId = sessionId ?? "";

  // Send SSE headers immediately
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  try {
    // Initialize OpenCode (lazy — first call starts the server)
    const opencode = await getOpencode();
    const { client } = opencode;

    // Create or resume session
    let ocSessionId: string;
    if (sessionId) {
      // Resume existing session
      ocSessionId = sessionId;
      logger.info(`[dispatch] resuming OpenCode session`, { sessionId });
    } else {
      const createRes = await client.session.create({
        body: {
          title: `holyshipper-${randomUUID().slice(0, 8)}`,
        },
      });
      if (createRes.error) {
        throw new Error(`Failed to create session: ${JSON.stringify(createRes.error)}`);
      }
      ocSessionId = createRes.data.id;
      logger.info(`[dispatch] created OpenCode session`, {
        sessionId: ocSessionId,
        title: createRes.data.title,
      });
    }
    resolvedSessionId = ocSessionId;
    sendSSE(res, { type: "session", sessionId: resolvedSessionId });

    // Subscribe to session events for real-time SSE streaming.
    // The event stream runs as a background async iterator while prompt() blocks.
    // This ensures tool_use and text events are forwarded to the caller in real-time.
    logger.info(`[dispatch] subscribing to OpenCode events`, { sessionId: ocSessionId });

    const eventAbort = new AbortController();
    const eventStream = await client.event.subscribe({ signal: eventAbort.signal });

    // Background event processor — runs concurrently with prompt()
    const eventLoop = (async () => {
      try {
        for await (const evt of eventStream.stream) {
          const event = evt as { type?: string; properties?: Record<string, unknown> };
          if (!event?.type) continue;

          const evtType = event.type;
          const props = event.properties ?? {};

          // Filter to our session only
          const evtSessionId = props.sessionID as string | undefined;
          if (evtSessionId && evtSessionId !== ocSessionId) continue;

          logger.debug(`[dispatch:sse] ${evtType}`, {
            sessionId: ocSessionId,
            propertyKeys: Object.keys(props),
          });

          // ── message.part.updated — tool invocations + text blocks ──
          if (evtType === "message.part.updated") {
            const part = props.part as Record<string, unknown> | undefined;
            if (!part) continue;
            const partType = part.type as string;

            if (partType === "tool") {
              const toolName = (part.tool as string) ?? "unknown";
              const state = part.state as Record<string, unknown> | undefined;
              const status = (state?.status as string) ?? "unknown";

              if (status === "pending" || status === "running") {
                toolUseCount++;
                logger.info(`[dispatch:sse] tool`, {
                  name: toolName,
                  status,
                  toolUseCount,
                  inputKeys: state?.input ? Object.keys(state.input as object) : [],
                });
                sendSSE(res, {
                  type: "tool_use",
                  name: toolName,
                  input: (state?.input as Record<string, unknown>) ?? {},
                });
              } else if (status === "completed") {
                logger.info(`[dispatch:sse] tool completed`, {
                  name: toolName,
                  title: (state?.title as string) ?? "",
                });
              } else if (status === "error") {
                logger.error(`[dispatch:sse] tool error`, {
                  name: toolName,
                  error: (state?.error as string) ?? "unknown",
                });
              }
            } else if (partType === "text") {
              const text = (part.text as string) ?? "";
              const delta = props.delta as string | undefined;
              const content = delta ?? text;
              if (content) {
                textBlockCount++;
                allText.push(content);
                logger.debug(`[dispatch:sse] text`, {
                  textBlockCount,
                  length: content.length,
                  preview: content.slice(0, 120),
                });
                sendSSE(res, { type: "text", text: content });
              }
            } else if (partType === "step-start" || partType === "step-finish") {
              logger.info(`[dispatch:sse] ${partType}`, { sessionId: ocSessionId });
              sendSSE(res, { type: "system", subtype: partType });
            }
          }

          // ── message.updated — track message lifecycle ──
          else if (evtType === "message.updated") {
            const info = props.info as Record<string, unknown> | undefined;
            if (info?.role === "assistant" && info?.time) {
              const time = info.time as Record<string, unknown>;
              if (time.completed) {
                logger.info(`[dispatch:sse] assistant message completed`, {
                  sessionId: ocSessionId,
                  messageId: info.id,
                  cost: info.cost,
                  tokens: info.tokens,
                  finish: info.finish,
                });
              }
            }
          }

          // ── session.status ──
          else if (evtType === "session.status") {
            const status = props.status as Record<string, unknown> | undefined;
            logger.info(`[dispatch:sse] session.status`, {
              sessionId: ocSessionId,
              status: status?.type ?? "unknown",
            });
          } else if (evtType === "session.idle") {
            logger.info(`[dispatch:sse] session.idle`, { sessionId: ocSessionId });
          }

          // ── session.error ──
          else if (evtType === "session.error") {
            logger.error(`[dispatch:sse] session.error`, {
              sessionId: ocSessionId,
              properties: props,
            });
          }

          // ── permission.updated — auto-accept so agent doesn't hang ──
          else if (evtType === "permission.updated") {
            const permissionId = props.id as string | undefined;
            logger.warn(`[dispatch:sse] permission requested`, {
              sessionId: ocSessionId,
              permissionId,
              tool: props.tool,
            });
            if (permissionId) {
              void autoAcceptPermission(client, ocSessionId, permissionId);
            }
          }

          // ── everything else — log for forensics ──
          else {
            logger.debug(`[dispatch:sse] unhandled event`, {
              type: evtType,
              sessionId: ocSessionId,
            });
          }
        }
      } catch (err) {
        // AbortError is expected when we cancel after prompt() returns
        if (err instanceof Error && err.name === "AbortError") return;
        logger.warn(`[dispatch:sse] event stream error`, {
          error: err instanceof Error ? err.message : String(err),
          sessionId: ocSessionId,
        });
      }
    })();

    // Send prompt — this blocks until the agent finishes
    logger.info(`[dispatch] sending prompt to OpenCode`, {
      sessionId: ocSessionId,
      model,
      promptLength: (data.prompt as string).length,
    });

    // OpenCode server expects providerID/modelID as flat body fields, not nested in a model object.
    const promptRes = await client.session.prompt({
      path: { id: ocSessionId },
      body: {
        providerID: model.providerID,
        modelID: model.modelID,
        parts: [{ type: "text" as const, text: data.prompt as string }],
        // biome-ignore lint/suspicious/noExplicitAny: SDK types expect nested model, server wants flat fields
      } as any,
    });

    if (promptRes.error) {
      logger.error(`[dispatch] prompt returned error`, {
        sessionId: ocSessionId,
        error: promptRes.error,
      });
      throw new Error(`Prompt failed: ${JSON.stringify(promptRes.error)}`);
    }

    const result = promptRes.data;

    // Process the final result
    const assistantInfo = result.info as Record<string, unknown> | undefined;
    const resultParts = result.parts as Array<Record<string, unknown>> | undefined;

    logger.info(`[dispatch] prompt completed`, {
      sessionId: ocSessionId,
      messageId: assistantInfo?.id,
      cost: assistantInfo?.cost,
      tokens: assistantInfo?.tokens,
      finish: assistantInfo?.finish,
      partCount: resultParts?.length ?? 0,
    });

    // Extract any text from result parts that weren't streamed via SSE
    for (const part of resultParts ?? []) {
      if (part.type === "text" && part.text) {
        const text = part.text as string;
        if (!allText.includes(text)) {
          textBlockCount++;
          allText.push(text);
          sendSSE(res, { type: "text", text });
        }
      }
    }

    // Stop the event stream and wait for the background loop to drain
    eventAbort.abort();
    await eventLoop;

    // Parse signal from all collected text
    const { signal, artifacts } = parseSignal(allText.join("\n"));
    const elapsed = Date.now() - startTime;

    logger.info(`[dispatch] result`, {
      signal,
      artifactKeys: Object.keys(artifacts),
      toolUseCount,
      textBlockCount,
      elapsedMs: elapsed,
      sessionId: ocSessionId,
      cost: assistantInfo?.cost,
    });

    sendSSE(res, {
      type: "result",
      subtype: "success",
      isError: false,
      stopReason: (assistantInfo?.finish as string) ?? "end_turn",
      costUsd: (assistantInfo?.cost as number) ?? null,
      signal,
      artifacts,
    });
  } catch (err) {
    const elapsed = Date.now() - startTime;
    logger.error(`[dispatch] OpenCode error`, {
      error: err instanceof Error ? err.message : String(err),
      stack: err instanceof Error ? err.stack : undefined,
      toolUseCount,
      textBlockCount,
      elapsedMs: elapsed,
    });
    sendSSE(res, {
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }

  const totalElapsed = Date.now() - startTime;
  logger.info(`[dispatch] stream complete`, {
    sessionId: resolvedSessionId,
    toolUseCount,
    textBlockCount,
    totalElapsedMs: totalElapsed,
  });
  res.end();
}

async function handleCredentials(req: IncomingMessage, res: ServerResponse): Promise<void> {
  logger.info(`[credentials] receiving credentials`);

  let body: string;
  try {
    body = await readBody(req);
  } catch (err) {
    logger.warn(`[credentials] body read failed`, { error: err instanceof Error ? err.message : String(err) });
    res.writeHead(400).end("Bad request");
    return;
  }

  let data: Record<string, unknown>;
  try {
    data = JSON.parse(body) as Record<string, unknown>;
    if (typeof data !== "object" || data === null || Array.isArray(data)) {
      throw new Error("Expected object");
    }
  } catch {
    logger.warn(`[credentials] invalid JSON body`);
    res.writeHead(400).end("Invalid JSON");
    return;
  }

  logger.info(`[credentials] credential types received`, { types: Object.keys(data) });

  const results: Record<string, boolean> = {};

  // Gateway service key — set env var so OpenCode provider picks it up
  if (data.gateway != null) {
    const gw = data.gateway as Record<string, unknown>;
    const key = typeof gw === "string" ? gw : (gw.key as string);
    if (key) {
      process.env.HOLYSHIP_GATEWAY_KEY = key;
      results.gateway = true;
      logger.info(`[credentials] gateway key set`, { keyLength: key.length, keyPrefix: key.slice(0, 8) });
    } else {
      logger.warn(`[credentials] gateway payload present but no key found`);
    }
  }

  // Gateway URL override
  if (typeof data.gatewayUrl === "string" && data.gatewayUrl) {
    process.env.HOLYSHIP_GATEWAY_URL = data.gatewayUrl;
    results.gatewayUrl = true;
    logger.info(`[credentials] gateway URL set`, { url: data.gatewayUrl });
  }

  // GitHub token — set env var so gh CLI and git pick it up
  if (data.github != null) {
    const gh = data.github as Record<string, unknown>;
    const token = typeof gh === "string" ? gh : (gh.token as string);
    if (token) {
      process.env.GH_TOKEN = token;
      process.env.GITHUB_TOKEN = token;
      results.github = true;
      logger.info(`[credentials] github token set`, { tokenLength: token.length });
    } else {
      logger.warn(`[credentials] github payload present but no token found`);
    }
  }

  if (Object.keys(results).length === 0) {
    logger.warn(`[credentials] no recognized credential types`, { receivedKeys: Object.keys(data) });
    res.writeHead(400).end("No recognized credential types (expected: gateway, github)");
    return;
  }

  logger.info(`[credentials] injection complete`, { results });
  res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify(results));
}

async function handleCheckout(req: IncomingMessage, res: ServerResponse): Promise<void> {
  logger.info(`[checkout] receiving checkout request`);

  let body: string;
  try {
    body = await readBody(req);
  } catch (err) {
    logger.warn(`[checkout] body read failed`, { error: err instanceof Error ? err.message : String(err) });
    res.writeHead(400).end("Bad request");
    return;
  }

  let data: Record<string, unknown>;
  try {
    data = JSON.parse(body) as Record<string, unknown>;
  } catch {
    logger.warn(`[checkout] invalid JSON body`);
    res.writeHead(400).end("Invalid JSON");
    return;
  }

  // Accept either `repos` (array) or `repo` (string, backwards compat)
  const repoList: string[] = Array.isArray(data.repos)
    ? (data.repos as string[]).map((r) => String(r).trim())
    : typeof data.repo === "string"
      ? [data.repo.trim()]
      : [];
  const branch = typeof data.branch === "string" ? data.branch.trim() || undefined : undefined;
  const entityId = typeof data.entityId === "string" ? data.entityId : undefined;

  if (repoList.length === 0) {
    logger.warn(`[checkout] missing required field: repo or repos`);
    res.writeHead(400).end("Missing required field: repo or repos");
    return;
  }

  // Reject repo values that start with '-' (after trim) to prevent flag injection into git/gh
  const flagLikeRepo = repoList.find((r) => r.startsWith("-"));
  if (flagLikeRepo) {
    logger.warn(`[checkout] rejected flag-like repo value`, { repo: flagLikeRepo });
    res.writeHead(400).end("Invalid repo value");
    return;
  }

  // Reject branch values that start with '-' to prevent flag injection into git checkout
  if (branch?.startsWith("-")) {
    logger.warn(`[checkout] rejected flag-like branch value`, { branch });
    res.writeHead(400).end("Invalid branch value");
    return;
  }

  // Sanitize entityId to prevent path traversal — only allow safe path segment characters
  const safeEntityId = entityId
    ? entityId.replace(/[^a-zA-Z0-9._-]/g, "_").replace(/^\.+/, "") || undefined
    : undefined;
  if (entityId && !safeEntityId) {
    logger.warn(`[checkout] rejected unsafe entityId`, { entityId });
    res.writeHead(400).end("Invalid entityId value");
    return;
  }

  // When entityId is provided, nest repos under WORKSPACE/entityId/
  const workspace = process.env.HOLYSHIPPER_WORKSPACE ?? WORKSPACE;
  const baseDir = safeEntityId ? join(workspace, safeEntityId) : workspace;

  logger.info(`[checkout] starting`, {
    repos: repoList,
    branch: branch ?? "(default)",
    baseDir,
    entityId: entityId ?? "(none)",
  });

  try {
    // Build env with GH token for gh/git auth
    const env = { ...process.env } as Record<string, string>;
    const ghToken = await resolveGhToken();
    if (ghToken) {
      env.GH_TOKEN = ghToken;
      env.GITHUB_TOKEN = ghToken;
      logger.info(`[checkout] GH token resolved`, { tokenLength: ghToken.length });
    } else {
      logger.warn(`[checkout] no GH token available`);
    }

    await mkdir(baseDir, { recursive: true });

    const worktrees: Record<string, string> = Object.create(null) as Record<string, string>;
    const targetBranch = branch ?? "main";

    for (const repo of repoList) {
      // Sanitize repoName: take the final path segment, strip leading dots (prevents `..`
      // traversal), replace unsafe chars to prevent prototype pollution
      const rawName = repo.split("/").pop() ?? repo;
      const repoName = rawName.replace(/[^a-zA-Z0-9._-]/g, "_").replace(/^\.+/, "") || "repo";
      const worktreePath = join(baseDir, repoName);

      // Clone if not already present
      if (!existsSync(worktreePath)) {
        logger.info(`[checkout] cloning repo`, { repo, worktreePath });
        const cloneStart = Date.now();
        // Use plain git clone for local paths; gh repo clone for remote OWNER/REPO refs
        const isLocalPath = repo.startsWith("/") || repo.startsWith("./") || repo.startsWith("../");
        if (isLocalPath) {
          await execFileAsync("git", ["clone", repo, worktreePath], { env });
        } else {
          await execFileAsync("gh", ["repo", "clone", repo, worktreePath], { env });
        }
        logger.info(`[checkout] clone complete`, { repo, elapsedMs: Date.now() - cloneStart });
      } else {
        logger.info(`[checkout] repo exists, fetching`, { repo, worktreePath });
        await execFileAsync("git", ["-C", worktreePath, "fetch", "origin"], { env });
        logger.info(`[checkout] fetch complete`, { repo });
      }

      // Create and checkout branch
      if (branch) {
        try {
          await execFileAsync("git", ["-C", worktreePath, "checkout", branch], { env });
          logger.info(`[checkout] checked out existing branch`, { repo, branch });
        } catch {
          await execFileAsync("git", ["-C", worktreePath, "checkout", "-b", branch], { env });
          logger.info(`[checkout] created new branch`, { repo, branch });
        }
      }

      worktrees[repoName] = worktreePath;
    }

    logger.info(`[checkout] complete`, { repos: repoList, branch: targetBranch, worktrees });

    // Return worktrees map. For single-repo backwards compat, also include flat fields.
    const firstRawName = repoList[0].split("/").pop() ?? repoList[0];
    const firstRepoName = firstRawName.replace(/[^a-zA-Z0-9._-]/g, "_").replace(/^\.+/, "") || "repo";
    res.writeHead(200, { "Content-Type": "application/json" }).end(
      JSON.stringify({
        worktrees,
        worktreePath: worktrees[firstRepoName],
        codebasePath: worktrees[firstRepoName],
        branch: targetBranch,
      }),
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logger.error(`[checkout] failed`, {
      repos: repoList,
      branch,
      error: msg,
      stack: err instanceof Error ? err.stack : undefined,
    });
    res.writeHead(500, { "Content-Type": "application/json" }).end(JSON.stringify({ error: msg }));
  }
}

async function handleGate(req: IncomingMessage, res: ServerResponse): Promise<void> {
  let body: string;
  try {
    body = await readBody(req);
  } catch (err) {
    logger.warn(`[gate] body read failed`, { error: err instanceof Error ? err.message : String(err) });
    res.writeHead(400).end("Bad request");
    return;
  }

  let data: unknown;
  try {
    data = JSON.parse(body);
  } catch {
    logger.warn(`[gate] invalid JSON body`);
    res.writeHead(400).end("Invalid JSON");
    return;
  }

  const request = data as Record<string, unknown>;
  if (typeof request.gateId !== "string" || !request.gateId) {
    res.writeHead(400).end("Missing gateId");
    return;
  }
  if (typeof request.entityId !== "string" || !request.entityId) {
    res.writeHead(400).end("Missing entityId");
    return;
  }
  if (typeof request.op !== "string" || !request.op) {
    res.writeHead(400).end("Missing op");
    return;
  }

  const rawParams = request.params;
  if (rawParams != null && (typeof rawParams !== "object" || Array.isArray(rawParams))) {
    res.writeHead(400).end("Invalid params");
    return;
  }

  const gateRequest: GateRequest = {
    gateId: request.gateId,
    entityId: request.entityId,
    op: request.op,
    params: (rawParams as Record<string, unknown>) ?? {},
    timeoutMs: typeof request.timeoutMs === "number" ? request.timeoutMs : undefined,
  };

  const result = await evaluateGate(gateRequest);
  res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify(result));
}

export function makeHandler(): RequestListener {
  return async (req, res) => {
    const { method, url } = req;

    if (method === "GET" && url === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ ok: true }));
      return;
    }

    logger.info(`[http] ${method} ${url}`);

    if (method === "POST" && url === "/credentials") {
      await handleCredentials(req, res);
      return;
    }

    if (method === "POST" && url === "/checkout") {
      await handleCheckout(req, res);
      return;
    }

    if (method === "POST" && url === "/dispatch") {
      await handleDispatch(req, res);
      return;
    }

    if (method === "POST" && url === "/gate") {
      await handleGate(req, res);
      return;
    }

    if (method === "GET" && url === "/gate/handlers") {
      res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ ops: listHandlers() }));
      return;
    }

    logger.warn(`[http] not found`, { method, url });
    res.writeHead(404).end("Not found");
  };
}

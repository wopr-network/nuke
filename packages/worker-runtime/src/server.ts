import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import type { IncomingMessage, RequestListener, ServerResponse } from "node:http";
import { homedir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { query } from "@anthropic-ai/claude-agent-sdk";
import { logger } from "./logger.js";
import { parseSignal } from "./parse-signal.js";
import type { DispatchRequest, NukeEvent } from "./types.js";

const execFileAsync = promisify(execFile);

const MODEL_MAP: Record<DispatchRequest["modelTier"], string> = {
  opus: "claude-opus-4-6",
  sonnet: "claude-sonnet-4-6",
  haiku: "claude-haiku-4-5",
};

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

function sendSSE(res: ServerResponse, event: NukeEvent): void {
  res.write(`data: ${JSON.stringify(event)}\n\n`);
}

/** Read GH token from /run/secrets, env, or credentials injection. */
async function resolveGhToken(): Promise<string | null> {
  // 1. Env var (set by credentials injection)
  if (process.env.GH_TOKEN) return process.env.GH_TOKEN;
  if (process.env.GITHUB_TOKEN) return process.env.GITHUB_TOKEN;
  // 2. Secrets file (mounted by NukeDispatcher; overridable in tests via GH_TOKEN_PATH_OVERRIDE)
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
  const sessionId =
    data.newSession === true ? undefined : typeof data.sessionId === "string" ? data.sessionId : undefined;

  const promptPreview = (data.prompt as string).slice(0, 200);
  logger.info(`[dispatch] received`, {
    modelTier,
    model: MODEL_MAP[modelTier],
    sessionId: sessionId ?? "(new)",
    promptLength: (data.prompt as string).length,
    promptPreview,
  });

  // For new sessions, generate a UUID and pass it to the SDK so both sides agree.
  // For resumes, the caller-supplied sessionId is used as-is.
  const resolvedSessionId = sessionId ?? randomUUID();

  const allText: string[] = [];
  const startTime = Date.now();
  let toolUseCount = 0;
  let textBlockCount = 0;

  // Send SSE headers immediately, then emit session event as first payload.
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  sendSSE(res, { type: "session", sessionId: resolvedSessionId });

  try {
    const env = Object.fromEntries(
      Object.entries(process.env).filter((e): e is [string, string] => e[1] !== undefined),
    );
    delete env.CLAUDECODE;

    const linearApiKey = env.LINEAR_API_KEY;
    const mcpServers = linearApiKey
      ? {
          "linear-server": {
            type: "stdio" as const,
            command: "npx",
            args: [
              "-y",
              "mcp-remote",
              "https://mcp.linear.app/mcp",
              "--header",
              `Authorization: Bearer ${linearApiKey}`,
            ],
            env,
          },
        }
      : undefined;

    if (mcpServers) {
      logger.info(`[dispatch] MCP servers configured`, { servers: Object.keys(mcpServers) });
    }

    logger.info(`[dispatch] starting SDK query`, { model: MODEL_MAP[modelTier], resolvedSessionId });

    for await (const message of query({
      prompt: data.prompt as string,
      options: {
        model: MODEL_MAP[modelTier],
        permissionMode: "bypassPermissions",
        // For new sessions, set the session ID so the SDK uses our UUID.
        // For resumes, pass the same ID via `resume`.
        ...(sessionId ? { resume: sessionId } : { sessionId: resolvedSessionId }),
        ...(mcpServers ? { mcpServers } : {}),
        env,
        /* v8 ignore next */
        stderr: (line: string) => logger.debug(`[sdk:stderr] ${line}`),
      },
    })) {
      const msg = message as { type: string };

      if (msg.type === "system") {
        const m = msg as { type: string; subtype: string; session_id?: string };
        logger.info(`[dispatch] system event`, { subtype: m.subtype, sessionId: resolvedSessionId });
        sendSSE(res, { type: "system", subtype: m.subtype });
      } else if (msg.type === "assistant") {
        const m = msg as { type: string; message: { content: unknown[] } };
        for (const block of m.message.content) {
          const b = block as { type: string };
          if (b.type === "tool_use") {
            const t = b as { type: string; name: string; input: Record<string, unknown> };
            toolUseCount++;
            logger.info(`[dispatch] tool_use`, {
              name: t.name,
              toolUseCount,
              inputKeys: Object.keys(t.input),
            });
            sendSSE(res, { type: "tool_use", name: t.name, input: t.input });
          } else if (b.type === "text") {
            const t = b as { type: string; text: string };
            if (t.text) {
              textBlockCount++;
              allText.push(t.text);
              logger.debug(`[dispatch] text block`, {
                textBlockCount,
                length: t.text.length,
                preview: t.text.slice(0, 120),
              });
              sendSSE(res, { type: "text", text: t.text });
            }
          }
        }
      } else if (msg.type === "result") {
        const m = msg as {
          type: string;
          subtype: string;
          is_error: boolean;
          stop_reason: string | null;
          total_cost_usd: number | null;
        };
        const { signal, artifacts } = parseSignal(allText.join("\n"));
        const elapsed = Date.now() - startTime;
        logger.info(`[dispatch] result`, {
          subtype: m.subtype,
          isError: m.is_error,
          stopReason: m.stop_reason,
          costUsd: m.total_cost_usd,
          signal,
          artifactKeys: Object.keys(artifacts),
          toolUseCount,
          textBlockCount,
          elapsedMs: elapsed,
        });
        sendSSE(res, {
          type: "result",
          subtype: m.subtype,
          isError: m.is_error,
          stopReason: m.stop_reason,
          costUsd: m.total_cost_usd,
          signal,
          artifacts,
        });
      }
    }
  } catch (err) {
    const elapsed = Date.now() - startTime;
    logger.error(`[dispatch] SDK error`, {
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

  // Claude credentials
  if (data.claude != null) {
    const claudeDir = join(homedir(), ".claude");
    await mkdir(claudeDir, { recursive: true });
    await writeFile(join(claudeDir, ".credentials.json"), JSON.stringify(data.claude), "utf-8");
    results.claude = true;
    logger.info(`[credentials] claude credentials written`, { path: join(claudeDir, ".credentials.json") });
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
    res.writeHead(400).end("No recognized credential types (expected: claude, github)");
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
    ? (data.repos as string[])
    : typeof data.repo === "string"
      ? [data.repo as string]
      : [];
  const branch = data.branch as string | undefined;
  const entityId = data.entityId as string | undefined;

  if (repoList.length === 0) {
    logger.warn(`[checkout] missing required field: repo or repos`);
    res.writeHead(400).end("Missing required field: repo or repos");
    return;
  }

  // Reject repo values that start with '-' to prevent flag injection into git/gh commands
  const flagLikeRepo = repoList.find((r) => r.startsWith("-"));
  if (flagLikeRepo) {
    logger.warn(`[checkout] rejected flag-like repo value`, { repo: flagLikeRepo });
    res.writeHead(400).end("Invalid repo value");
    return;
  }

  // When entityId is provided, nest repos under WORKSPACE/entityId/
  const workspace = process.env.NUKE_WORKSPACE ?? WORKSPACE;
  const baseDir = entityId ? join(workspace, entityId) : workspace;

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
      // Sanitize repoName: strip leading dots/slashes to get the final path segment,
      // replace non-alphanumeric chars (except - and _) to prevent prototype pollution
      const rawName = repo.split("/").pop() ?? repo;
      const repoName = rawName.replace(/[^a-zA-Z0-9._-]/g, "_") || "repo";
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
    const firstRepoName = firstRawName.replace(/[^a-zA-Z0-9._-]/g, "_") || "repo";
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

    logger.warn(`[http] not found`, { method, url });
    res.writeHead(404).end("Not found");
  };
}

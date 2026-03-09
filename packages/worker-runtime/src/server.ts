import { randomUUID } from "node:crypto";
import type { IncomingMessage, RequestListener, ServerResponse } from "node:http";
import { query } from "@anthropic-ai/claude-agent-sdk";
import { parseSignal } from "./parse-signal.js";
import type { NukeEvent, DispatchRequest } from "./types.js";

const MODEL_MAP: Record<DispatchRequest["modelTier"], string> = {
  opus: "claude-opus-4-6",
  sonnet: "claude-sonnet-4-6",
  haiku: "claude-haiku-4-5",
};

const MAX_BODY_SIZE = 1024 * 1024;

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let totalSize = 0;
    req.on("data", (chunk: Buffer) => {
      totalSize += chunk.length;
      if (totalSize > MAX_BODY_SIZE) {
        reject(new Error("Request body too large"));
        req.destroy();
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

async function handleDispatch(req: IncomingMessage, res: ServerResponse): Promise<void> {
  let body: string;
  try {
    body = await readBody(req);
  } catch {
    res.writeHead(400).end("Bad request");
    return;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    res.writeHead(400).end("Invalid JSON");
    return;
  }

  const data = parsed as Record<string, unknown>;
  if (typeof data.prompt !== "string" || !data.prompt) {
    res.writeHead(400).end("Missing prompt");
    return;
  }

  const modelTier = (data.modelTier as DispatchRequest["modelTier"]) ?? "sonnet";
  const sessionId =
    data.newSession === true ? undefined : typeof data.sessionId === "string" ? data.sessionId : undefined;

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  // First event: session
  const newSessionId = sessionId ?? randomUUID();
  sendSSE(res, { type: "session", sessionId: newSessionId });

  const allText: string[] = [];

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
            args: ["-y", "mcp-remote", "https://mcp.linear.app/mcp", "--header", `Authorization: Bearer ${linearApiKey}`],
            env,
          },
        }
      : undefined;

    for await (const message of query({
      prompt: data.prompt as string,
      options: {
        model: MODEL_MAP[modelTier],
        permissionMode: "bypassPermissions",
        ...(sessionId ? { resume: sessionId } : {}),
        ...(mcpServers ? { mcpServers } : {}),
        env,
        stderr: (line: string) => process.stderr.write(`[sdk] ${line}\n`),
      },
    })) {
      const msg = message as { type: string };

      if (msg.type === "system") {
        const m = msg as { type: string; subtype: string };
        sendSSE(res, { type: "system", subtype: m.subtype });
      } else if (msg.type === "assistant") {
        const m = msg as { type: string; message: { content: unknown[] } };
        for (const block of m.message.content) {
          const b = block as { type: string };
          if (b.type === "tool_use") {
            const t = b as { type: string; name: string; input: Record<string, unknown> };
            sendSSE(res, { type: "tool_use", name: t.name, input: t.input });
          } else if (b.type === "text") {
            const t = b as { type: string; text: string };
            if (t.text) {
              allText.push(t.text);
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
    sendSSE(res, {
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }

  res.end();
}

export function makeHandler(): RequestListener {
  return async (req, res) => {
    const { method, url } = req;

    if (method === "GET" && url === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ ok: true }));
      return;
    }

    if (method === "POST" && url === "/dispatch") {
      await handleDispatch(req, res);
      return;
    }

    res.writeHead(404).end("Not found");
  };
}

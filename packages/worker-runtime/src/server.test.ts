import { createServer, type Server } from "node:http";
import { AddressInfo } from "node:net";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { query } from "@anthropic-ai/claude-agent-sdk";

vi.mock("@anthropic-ai/claude-agent-sdk", () => ({
  query: vi.fn(),
}));

const mockQuery = vi.mocked(query);

async function* makeStream(messages: object[]) {
  for (const m of messages) yield m;
}

async function startServer(): Promise<{ url: string; server: Server }> {
  const { makeHandler } = await import("./server.js");
  const handler = makeHandler();
  const server = createServer(handler);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return { url: `http://127.0.0.1:${port}`, server };
}

function stopServer(server: Server): Promise<void> {
  return new Promise((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
}

async function parseSSE(res: Response): Promise<object[]> {
  const text = await res.text();
  return text
    .split("\n")
    .filter((l: string) => l.startsWith("data:"))
    .map((l: string) => JSON.parse(l.slice(5)) as object);
}

let url: string;
let server: Server;

beforeEach(async () => {
  vi.resetModules();
  mockQuery.mockReset();
  ({ url, server } = await startServer());
});

afterEach(async () => {
  await stopServer(server);
});

describe("POST /dispatch", () => {
  it("returns 400 for missing prompt", async () => {
    const res = await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ modelTier: "haiku" }),
    });
    expect(res.status).toBe(400);
  });

  it("streams session event as first SSE event", async () => {
    mockQuery.mockReturnValue(
      makeStream([
        { type: "result", subtype: "success", is_error: false, total_cost_usd: 0.001, stop_reason: "end_turn" },
      ]) as ReturnType<typeof query>,
    );

    const res = await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "do work", modelTier: "haiku" }),
    });

    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");

    const events = await parseSSE(res);
    const first = events[0] as { type: string; sessionId: string };
    expect(first.type).toBe("session");
    expect(typeof first.sessionId).toBe("string");
    expect(first.sessionId.length).toBeGreaterThan(0);
  });

  it("streams tool_use events", async () => {
    mockQuery.mockReturnValue(
      makeStream([
        {
          type: "assistant",
          message: {
            content: [{ type: "tool_use", name: "Read", input: { file_path: "/foo.ts" } }],
          },
        },
        { type: "result", subtype: "success", is_error: false, total_cost_usd: 0, stop_reason: "end_turn" },
      ]) as ReturnType<typeof query>,
    );

    const res = await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "do work", modelTier: "haiku" }),
    });

    const events = await parseSSE(res);
    const toolUse = events.find((e) => (e as { type: string }).type === "tool_use") as {
      type: string;
      name: string;
      input: Record<string, unknown>;
    };
    expect(toolUse).toBeDefined();
    expect(toolUse.name).toBe("Read");
    expect(toolUse.input).toEqual({ file_path: "/foo.ts" });
  });

  it("streams text events", async () => {
    mockQuery.mockReturnValue(
      makeStream([
        {
          type: "assistant",
          message: { content: [{ type: "text", text: "thinking hard" }] },
        },
        { type: "result", subtype: "success", is_error: false, total_cost_usd: 0, stop_reason: "end_turn" },
      ]) as ReturnType<typeof query>,
    );

    const res = await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "do work", modelTier: "haiku" }),
    });

    const events = await parseSSE(res);
    const textEvent = events.find((e) => (e as { type: string }).type === "text") as { type: string; text: string };
    expect(textEvent?.text).toBe("thinking hard");
  });

  it("streams result event with parsed signal", async () => {
    mockQuery.mockReturnValue(
      makeStream([
        {
          type: "assistant",
          message: {
            content: [{ type: "text", text: "PR created: https://github.com/wopr-network/radar/pull/42" }],
          },
        },
        { type: "result", subtype: "success", is_error: false, total_cost_usd: 0.005, stop_reason: "end_turn" },
      ]) as ReturnType<typeof query>,
    );

    const res = await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "do work", modelTier: "sonnet" }),
    });

    const events = await parseSSE(res);
    const result = events.find((e) => (e as { type: string }).type === "result") as {
      type: string;
      signal: string;
      artifacts: Record<string, unknown>;
      costUsd: number;
      isError: boolean;
    };
    expect(result?.signal).toBe("pr_created");
    expect(result?.artifacts).toMatchObject({ prNumber: 42 });
    expect(result?.costUsd).toBe(0.005);
    expect(result?.isError).toBe(false);
  });

  it("reuses session when sessionId provided", async () => {
    mockQuery.mockReturnValue(
      makeStream([
        { type: "result", subtype: "success", is_error: false, total_cost_usd: 0, stop_reason: "end_turn" },
      ]) as ReturnType<typeof query>,
    );

    await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "do work", modelTier: "haiku", sessionId: "existing-session-abc" }),
    });

    expect(mockQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        options: expect.objectContaining({ resume: "existing-session-abc" }),
      }),
    );
  });

  it("starts fresh session when newSession=true even if sessionId provided", async () => {
    mockQuery.mockReturnValue(
      makeStream([
        { type: "result", subtype: "success", is_error: false, total_cost_usd: 0, stop_reason: "end_turn" },
      ]) as ReturnType<typeof query>,
    );

    await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "do work", modelTier: "haiku", sessionId: "old-session", newSession: true }),
    });

    const callOpts = mockQuery.mock.calls[0][0].options as Record<string, unknown>;
    expect(callOpts.resume).toBeUndefined();
  });

  it("streams error event on query failure", async () => {
    mockQuery.mockReturnValue(
      (async function* () {
        throw new Error("SDK exploded");
      })() as unknown as ReturnType<typeof query>,
    );

    const res = await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "do work", modelTier: "haiku" }),
    });

    const events = await parseSSE(res);
    const error = events.find((e) => (e as { type: string }).type === "error") as { type: string; message: string };
    expect(error?.message).toContain("SDK exploded");
  });
});

describe("GET /health", () => {
  it("returns 200", async () => {
    const res = await fetch(`${url}/health`);
    expect(res.status).toBe(200);
  });
});

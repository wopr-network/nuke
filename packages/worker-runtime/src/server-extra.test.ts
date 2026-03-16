import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { createOpencode } from "@opencode-ai/sdk";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@opencode-ai/sdk", () => ({
  createOpencode: vi.fn(),
}));

const mockSessionCreate = vi.fn();
const mockSessionPrompt = vi.fn();
const mockEventSubscribe = vi.fn();

vi.mocked(createOpencode).mockResolvedValue({
  client: {
    session: { create: mockSessionCreate, prompt: mockSessionPrompt },
    event: { subscribe: mockEventSubscribe },
  } as never,
  server: { url: "http://localhost:4096", close: vi.fn() },
});

function mockSuccessfulDispatch(text = "done") {
  mockSessionCreate.mockResolvedValue({
    data: { id: "session-abc", title: "test" },
    error: undefined,
  });
  mockSessionPrompt.mockResolvedValue({
    data: {
      info: { id: "msg-1", cost: 0, finish: "end_turn", tokens: {} },
      parts: [{ type: "text", text }],
    },
    error: undefined,
  });
  mockEventSubscribe.mockResolvedValue({
    stream: (async function* () {})(),
  });
}

async function startServer(): Promise<{ url: string; server: Server }> {
  const { makeHandler } = await import("./server.js");
  const server = createServer(makeHandler());
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
  mockSessionCreate.mockReset();
  mockSessionPrompt.mockReset();
  mockEventSubscribe.mockReset();
  ({ url, server } = await startServer());
});

afterEach(async () => {
  await stopServer(server);
});

describe("POST /dispatch — model tier mapping", () => {
  it.each([
    ["opus", "anthropic/claude-opus-4-6"],
    ["sonnet", "anthropic/claude-sonnet-4-6"],
    ["haiku", "anthropic/claude-haiku-4-5"],
  ])("maps tier %s to model %s", async (tier, expectedModelID) => {
    mockSuccessfulDispatch();

    await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "work", modelTier: tier }),
    });

    expect(mockSessionPrompt).toHaveBeenCalledWith(
      expect.objectContaining({
        body: expect.objectContaining({
          modelID: expectedModelID,
          providerID: "holyship",
        }),
      }),
    );
  });

  it("defaults to sonnet for missing modelTier", async () => {
    mockSuccessfulDispatch();

    await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "work" }),
    });

    expect(mockSessionPrompt).toHaveBeenCalledWith(
      expect.objectContaining({
        body: expect.objectContaining({
          modelID: "anthropic/claude-sonnet-4-6",
          providerID: "holyship",
        }),
      }),
    );
  });
});

describe("POST /dispatch — request validation", () => {
  it("returns 400 for invalid JSON", async () => {
    const res = await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "not json {{{",
    });
    expect(res.status).toBe(400);
  });

  it("returns 400 when body exceeds 1MB", async () => {
    const bigBody = "x".repeat(1024 * 1024 + 1);
    const res = await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: bigBody,
    });
    expect(res.status).toBe(400);
  });

  it("returns 400 for empty prompt string", async () => {
    const res = await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "", modelTier: "haiku" }),
    });
    expect(res.status).toBe(400);
  });
});

describe("POST /dispatch — session ID handling", () => {
  it("creates new session and returns UUID", async () => {
    mockSuccessfulDispatch();

    const res = await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "work", modelTier: "haiku" }),
    });

    const events = await parseSSE(res);
    const session = events[0] as { type: string; sessionId: string };
    expect(session.type).toBe("session");
    expect(session.sessionId).toBe("session-abc");
  });

  it("reuses session when sessionId provided", async () => {
    mockSuccessfulDispatch();

    const res = await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "work", modelTier: "haiku", sessionId: "existing-session" }),
    });

    // Should NOT create a new session
    expect(mockSessionCreate).not.toHaveBeenCalled();
    // Should prompt with the existing session ID
    expect(mockSessionPrompt).toHaveBeenCalledWith(expect.objectContaining({ path: { id: "existing-session" } }));

    const events = await parseSSE(res);
    const session = events[0] as { type: string; sessionId: string };
    expect(session.sessionId).toBe("existing-session");
  });

  it("creates new session when newSession=true even if sessionId provided", async () => {
    mockSuccessfulDispatch();

    const res = await fetch(`${url}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "work", modelTier: "haiku", sessionId: "old-id", newSession: true }),
    });

    // Should create a NEW session
    expect(mockSessionCreate).toHaveBeenCalled();
    const events = await parseSSE(res);
    const session = events[0] as { type: string; sessionId: string };
    expect(session.sessionId).not.toBe("old-id");
  });
});

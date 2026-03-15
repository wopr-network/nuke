import { execFile } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { query } from "@anthropic-ai/claude-agent-sdk";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const execFileAsync = promisify(execFile);

// Redirect homedir() to a temp directory so tests never touch real credentials
const fakeHome = mkdtempSync(join(tmpdir(), "holyshipper-test-"));
vi.mock("node:os", async () => {
  const actual = await vi.importActual<typeof import("node:os")>("node:os");
  return { ...actual, homedir: () => fakeHome };
});

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
        yield { type: "system", subtype: "init" };
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

describe("POST /credentials", () => {
  it("writes claude credentials to ~/.claude/.credentials.json", async () => {
    const creds = { oauthToken: "test-token-123", expiresAt: "2026-12-31" };
    const res = await fetch(`${url}/credentials`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claude: creds }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, boolean>;
    expect(body.claude).toBe(true);

    const written = await readFile(join(fakeHome, ".claude", ".credentials.json"), "utf-8");
    expect(JSON.parse(written)).toEqual(creds);
  });

  it("sets GH_TOKEN env var from github credentials", async () => {
    const originalGh = process.env.GH_TOKEN;
    const originalGithub = process.env.GITHUB_TOKEN;

    const res = await fetch(`${url}/credentials`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ github: { token: "ghp_test123" } }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, boolean>;
    expect(body.github).toBe(true);
    expect(process.env.GH_TOKEN).toBe("ghp_test123");
    expect(process.env.GITHUB_TOKEN).toBe("ghp_test123");

    // Restore
    process.env.GH_TOKEN = originalGh;
    process.env.GITHUB_TOKEN = originalGithub;
  });

  it("accepts both claude and github in one call", async () => {
    const res = await fetch(`${url}/credentials`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        claude: { oauthToken: "tok" },
        github: { token: "ghp_both" },
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, boolean>;
    expect(body.claude).toBe(true);
    expect(body.github).toBe(true);
  });

  it("returns 200 but skips token when github object has no token field", async () => {
    const res = await fetch(`${url}/credentials`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claude: { oauth_token: "x", refresh_token: "y" }, github: {} }),
    });
    // claude cred is recognized, so not a 400
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, boolean>;
    expect(body.github).toBeUndefined();
  });

  it("returns 400 for no recognized credential types", async () => {
    const res = await fetch(`${url}/credentials`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ unknown: "stuff" }),
    });
    expect(res.status).toBe(400);
  });

  it("returns 400 for oversized body", async () => {
    const res = await fetch(`${url}/credentials`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "x".repeat(1024 * 1024 + 1),
    });
    expect(res.status).toBe(400);
  });

  it("returns 400 for invalid JSON", async () => {
    const res = await fetch(`${url}/credentials`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "not json",
    });
    expect(res.status).toBe(400);
  });

  it("returns 400 for non-object JSON", async () => {
    const res = await fetch(`${url}/credentials`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([1, 2, 3]),
    });
    expect(res.status).toBe(400);
  });
});

describe("POST /checkout", () => {
  it("returns 400 for missing repo", async () => {
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ branch: "wop-123" }),
    });
    expect(res.status).toBe(400);
  });

  it("returns 400 for invalid JSON", async () => {
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "not json",
    });
    expect(res.status).toBe(400);
  });

  it("returns 500 when workspace is not writable (default /workspace path)", async () => {
    // No HOLYSHIPPER_WORKSPACE set — falls back to /workspace which is not writable in test env
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: "nonexistent/repo-that-will-fail", branch: "test-branch" }),
    });
    expect(res.status).toBe(500);
    const body = (await res.json()) as { error: string };
    expect(body.error).toBeDefined();
  });

  it("returns 500 when gh clone fails for nonexistent remote repo", async () => {
    const wsDir = mkdtempSync(join(tmpdir(), "holyshipper-ws-"));
    process.env.HOLYSHIPPER_WORKSPACE = wsDir;
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: "nonexistent/repo-that-will-fail", branch: "test-branch" }),
    });
    delete process.env.HOLYSHIPPER_WORKSPACE;
    expect(res.status).toBe(500);
    const body = (await res.json()) as { error: string };
    expect(body.error).toBeDefined();
  });

  it("returns 400 for empty body", async () => {
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "",
    });
    expect(res.status).toBe(400);
  });

  it("returns 400 for oversized body", async () => {
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "x".repeat(1024 * 1024 + 1),
    });
    expect(res.status).toBe(400);
  });

  it("clones without GH token when env vars absent", async () => {
    const origGH = process.env.GH_TOKEN;
    const origGithub = process.env.GITHUB_TOKEN;
    delete process.env.GH_TOKEN;
    delete process.env.GITHUB_TOKEN;

    const tmpDir = mkdtempSync(join(tmpdir(), "holyshipper-notoken-"));
    const bareDir = join(tmpDir, "test-repo");
    const workDir = join(tmpDir, "work");
    await execFileAsync("git", ["init", "--bare", bareDir]);
    await execFileAsync("git", ["clone", bareDir, workDir]);
    await execFileAsync("git", ["-C", workDir, "config", "user.email", "test@test.com"]);
    await execFileAsync("git", ["-C", workDir, "config", "user.name", "Test"]);
    await execFileAsync("git", ["-C", workDir, "commit", "--allow-empty", "-m", "init"]);
    await execFileAsync("git", ["-C", workDir, "push", "origin", "HEAD:main"]);

    const wsDir = mkdtempSync(join(tmpdir(), "holyshipper-ws-"));
    process.env.HOLYSHIPPER_WORKSPACE = wsDir;
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: bareDir }),
    });
    delete process.env.HOLYSHIPPER_WORKSPACE;
    process.env.GH_TOKEN = origGH;
    process.env.GITHUB_TOKEN = origGithub;

    expect(res.status).toBe(200);
  });

  it("returns 400 for flag-like repo value (command injection prevention)", async () => {
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: "--upload-pack=evil" }),
    });
    expect(res.status).toBe(400);
  });

  it("returns 400 for flag-like repo in repos array", async () => {
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repos: ["valid/repo", "--malicious"] }),
    });
    expect(res.status).toBe(400);
  });

  it("returns 400 for flag-like repo with leading whitespace (bypass attempt)", async () => {
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: "  --upload-pack=evil" }),
    });
    expect(res.status).toBe(400);
  });

  it("returns 400 for flag-like branch value", async () => {
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: "owner/some-repo", branch: "--force" }),
    });
    expect(res.status).toBe(400);
  });

  it("returns 400 when repo field is missing but other fields present", async () => {
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ branch: "feat-x", other: "data" }),
    });
    expect(res.status).toBe(400);
    const text = await res.text();
    expect(text).toContain("repo");
  });

  it("clones a local repo and returns worktrees map", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "holyshipper-clone-"));
    const bareDir = join(tmpDir, "test-repo");
    const workDir = join(tmpDir, "work");

    await execFileAsync("git", ["init", "--bare", bareDir]);
    await execFileAsync("git", ["clone", bareDir, workDir]);
    await execFileAsync("git", ["-C", workDir, "config", "user.email", "test@test.com"]);
    await execFileAsync("git", ["-C", workDir, "config", "user.name", "Test"]);
    await execFileAsync("git", ["-C", workDir, "commit", "--allow-empty", "-m", "init"]);
    await execFileAsync("git", ["-C", workDir, "push", "origin", "HEAD:main"]);

    const wsDir = mkdtempSync(join(tmpdir(), "holyshipper-ws-"));
    process.env.HOLYSHIPPER_WORKSPACE = wsDir;

    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: bareDir, branch: "feat-test" }),
    });
    delete process.env.HOLYSHIPPER_WORKSPACE;

    expect(res.status).toBe(200);
    const body = (await res.json()) as { worktrees: Record<string, string>; worktreePath: string; branch: string };
    expect(body.branch).toBe("feat-test");
    expect(body.worktreePath).toBeDefined();
    expect(typeof body.worktrees).toBe("object");
  });

  it("fetches when worktree already exists and checks out existing branch", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "holyshipper-fetch-"));
    const bareDir = join(tmpDir, "test-repo");
    const workDir = join(tmpDir, "work");

    await execFileAsync("git", ["init", "--bare", bareDir]);
    await execFileAsync("git", ["clone", bareDir, workDir]);
    await execFileAsync("git", ["-C", workDir, "config", "user.email", "test@test.com"]);
    await execFileAsync("git", ["-C", workDir, "config", "user.name", "Test"]);
    await execFileAsync("git", ["-C", workDir, "commit", "--allow-empty", "-m", "init"]);
    await execFileAsync("git", ["-C", workDir, "push", "origin", "HEAD:main"]);
    // Create the target branch in the remote
    await execFileAsync("git", ["-C", workDir, "checkout", "-b", "existing-branch"]);
    await execFileAsync("git", ["-C", workDir, "push", "origin", "existing-branch"]);

    const wsDir = mkdtempSync(join(tmpdir(), "holyshipper-ws-"));
    // Pre-clone into the workspace so existsSync returns true
    await execFileAsync("git", ["clone", bareDir, join(wsDir, "test-repo")]);

    process.env.HOLYSHIPPER_WORKSPACE = wsDir;
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: bareDir, branch: "existing-branch" }),
    });
    delete process.env.HOLYSHIPPER_WORKSPACE;

    expect(res.status).toBe(200);
    const body = (await res.json()) as { branch: string };
    expect(body.branch).toBe("existing-branch");
  });

  it("clones a ../relative-path repo", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "holyshipper-dotdot-"));
    const bareDir = join(tmpDir, "test-repo");
    const workDir = join(tmpDir, "work");

    await execFileAsync("git", ["init", "--bare", bareDir]);
    await execFileAsync("git", ["clone", bareDir, workDir]);
    await execFileAsync("git", ["-C", workDir, "config", "user.email", "test@test.com"]);
    await execFileAsync("git", ["-C", workDir, "config", "user.name", "Test"]);
    await execFileAsync("git", ["-C", workDir, "commit", "--allow-empty", "-m", "init"]);
    await execFileAsync("git", ["-C", workDir, "push", "origin", "HEAD:main"]);

    const wsDir = mkdtempSync(join(tmpdir(), "holyshipper-ws-"));
    process.env.HOLYSHIPPER_WORKSPACE = wsDir;

    // Use a path starting with ../  — server should use git clone not gh
    const nested = join(tmpDir, "nested");
    const { mkdirSync } = await import("node:fs");
    mkdirSync(nested, { recursive: true });
    const origCwd = process.cwd();
    process.chdir(nested);
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: "../test-repo" }),
    });
    process.chdir(origCwd);
    delete process.env.HOLYSHIPPER_WORKSPACE;

    expect(res.status).toBe(200);
  });

  it("clones a ./relative-path repo", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "holyshipper-rel-"));
    const bareDir = join(tmpDir, "test-repo");
    const workDir = join(tmpDir, "work");

    await execFileAsync("git", ["init", "--bare", bareDir]);
    await execFileAsync("git", ["clone", bareDir, workDir]);
    await execFileAsync("git", ["-C", workDir, "config", "user.email", "test@test.com"]);
    await execFileAsync("git", ["-C", workDir, "config", "user.name", "Test"]);
    await execFileAsync("git", ["-C", workDir, "commit", "--allow-empty", "-m", "init"]);
    await execFileAsync("git", ["-C", workDir, "push", "origin", "HEAD:main"]);

    const wsDir = mkdtempSync(join(tmpdir(), "holyshipper-ws-"));
    process.env.HOLYSHIPPER_WORKSPACE = wsDir;

    // Use a relative path starting with ./ — server should use git clone not gh
    const origCwd = process.cwd();
    process.chdir(tmpDir);
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: "./test-repo" }),
    });
    process.chdir(origCwd);
    delete process.env.HOLYSHIPPER_WORKSPACE;

    expect(res.status).toBe(200);
  });

  it("reads GH token from file when env vars absent", async () => {
    const origGH = process.env.GH_TOKEN;
    const origGithub = process.env.GITHUB_TOKEN;
    delete process.env.GH_TOKEN;
    delete process.env.GITHUB_TOKEN;

    // Write a fake token to a temp file and point the server at it
    const tokenFile = join(mkdtempSync(join(tmpdir(), "holyshipper-tok-")), "gh-token");
    const { writeFileSync } = await import("node:fs");
    writeFileSync(tokenFile, "fake-token-from-file");
    process.env.GH_TOKEN_PATH_OVERRIDE = tokenFile;

    const tmpDir = mkdtempSync(join(tmpdir(), "holyshipper-tok-repo-"));
    const bareDir = join(tmpDir, "test-repo");
    const workDir = join(tmpDir, "work");
    await execFileAsync("git", ["init", "--bare", bareDir]);
    await execFileAsync("git", ["clone", bareDir, workDir]);
    await execFileAsync("git", ["-C", workDir, "config", "user.email", "test@test.com"]);
    await execFileAsync("git", ["-C", workDir, "config", "user.name", "Test"]);
    await execFileAsync("git", ["-C", workDir, "commit", "--allow-empty", "-m", "init"]);
    await execFileAsync("git", ["-C", workDir, "push", "origin", "HEAD:main"]);

    const wsDir = mkdtempSync(join(tmpdir(), "holyshipper-ws-"));
    process.env.HOLYSHIPPER_WORKSPACE = wsDir;
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: bareDir }),
    });
    delete process.env.HOLYSHIPPER_WORKSPACE;
    delete process.env.GH_TOKEN_PATH_OVERRIDE;
    process.env.GH_TOKEN = origGH;
    process.env.GITHUB_TOKEN = origGithub;

    expect(res.status).toBe(200);
  });

  it("clones with entityId and nests under entityId subdir", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "holyshipper-entity-"));
    const bareDir = join(tmpDir, "test-repo");
    const workDir = join(tmpDir, "work");

    await execFileAsync("git", ["init", "--bare", bareDir]);
    await execFileAsync("git", ["clone", bareDir, workDir]);
    await execFileAsync("git", ["-C", workDir, "config", "user.email", "test@test.com"]);
    await execFileAsync("git", ["-C", workDir, "config", "user.name", "Test"]);
    await execFileAsync("git", ["-C", workDir, "commit", "--allow-empty", "-m", "init"]);
    await execFileAsync("git", ["-C", workDir, "push", "origin", "HEAD:main"]);

    const wsDir = mkdtempSync(join(tmpdir(), "holyshipper-ws-"));
    process.env.HOLYSHIPPER_WORKSPACE = wsDir;

    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: bareDir, entityId: "entity-123" }),
    });
    delete process.env.HOLYSHIPPER_WORKSPACE;

    expect(res.status).toBe(200);
    const body = (await res.json()) as { worktreePath: string };
    expect(body.worktreePath).toContain("entity-123");
  });

  it("rejects entityId that reduces to empty after sanitization", async () => {
    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: "owner/repo", entityId: "..." }),
    });
    expect(res.status).toBe(400);
    expect(await res.text()).toBe("Invalid entityId value");
  });

  it("clones multiple local repos and returns worktrees map", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "holyshipper-multi-"));

    const makeRepo = async (name: string) => {
      const bareDir = join(tmpDir, name);
      const workDir = join(tmpDir, `${name}-work`);
      await execFileAsync("git", ["init", "--bare", bareDir]);
      await execFileAsync("git", ["clone", bareDir, workDir]);
      await execFileAsync("git", ["-C", workDir, "config", "user.email", "test@test.com"]);
      await execFileAsync("git", ["-C", workDir, "config", "user.name", "Test"]);
      await execFileAsync("git", ["-C", workDir, "commit", "--allow-empty", "-m", "init"]);
      await execFileAsync("git", ["-C", workDir, "push", "origin", "HEAD:main"]);
      return bareDir;
    };

    const [repo1, repo2] = await Promise.all([makeRepo("repo-a"), makeRepo("repo-b")]);
    const wsDir = mkdtempSync(join(tmpdir(), "holyshipper-ws-"));
    process.env.HOLYSHIPPER_WORKSPACE = wsDir;

    const res = await fetch(`${url}/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repos: [repo1, repo2], branch: "feat-multi" }),
    });
    delete process.env.HOLYSHIPPER_WORKSPACE;

    expect(res.status).toBe(200);
    const body = (await res.json()) as { worktrees: Record<string, string>; branch: string };
    expect(body.branch).toBe("feat-multi");
    expect(Object.keys(body.worktrees).length).toBe(2);
  });
});

describe("POST /credentials — github string shorthand", () => {
  it("accepts github as bare string token", async () => {
    const originalGh = process.env.GH_TOKEN;
    const originalGithub = process.env.GITHUB_TOKEN;

    const res = await fetch(`${url}/credentials`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ github: "ghp_shorthand_token" }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, boolean>;
    expect(body.github).toBe(true);
    expect(process.env.GH_TOKEN).toBe("ghp_shorthand_token");

    process.env.GH_TOKEN = originalGh;
    process.env.GITHUB_TOKEN = originalGithub;
  });
});

describe("GET /health", () => {
  it("returns 200", async () => {
    const res = await fetch(`${url}/health`);
    expect(res.status).toBe(200);
  });
});

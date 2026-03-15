import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { clearHandlers } from "../gates.js";
import { ciStatus, commentExists, prCapacity, prStatus, registerGitHubHandlers } from "./github.js";

// Mock fetch globally
const mockFetch = vi.fn();

beforeAll(() => {
  vi.stubGlobal("fetch", mockFetch);
});

afterAll(() => {
  vi.unstubAllGlobals();
});

beforeEach(() => {
  process.env.GH_TOKEN = "test-token-123";
  mockFetch.mockReset();
});

afterEach(() => {
  delete process.env.GH_TOKEN;
  delete process.env.GITHUB_TOKEN;
  clearHandlers();
});

const ctx = { entityId: "e-1" };

describe("vcs.ci_status", () => {
  it("returns passed when all checks succeed", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        check_runs: [
          { status: "completed", conclusion: "success" },
          { status: "completed", conclusion: "skipped" },
        ],
      }),
    });

    const result = await ciStatus("vcs.ci_status", { repo: "org/repo", ref: "abc123" }, ctx);
    expect(result.outcome).toBe("passed");
  });

  it("returns pending when checks still running", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        check_runs: [
          { status: "completed", conclusion: "success" },
          { status: "in_progress", conclusion: null },
        ],
      }),
    });

    const result = await ciStatus("vcs.ci_status", { repo: "org/repo", ref: "abc123" }, ctx);
    expect(result.outcome).toBe("pending");
  });

  it("returns failed when a check fails", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        check_runs: [
          { status: "completed", conclusion: "failure" },
        ],
      }),
    });

    const result = await ciStatus("vcs.ci_status", { repo: "org/repo", ref: "abc123" }, ctx);
    expect(result.outcome).toBe("failed");
  });

  it("returns error on missing params", async () => {
    const result = await ciStatus("vcs.ci_status", { repo: "org/repo" }, ctx);
    expect(result.outcome).toBe("error");
  });

  it("returns error on API failure", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 403 });

    const result = await ciStatus("vcs.ci_status", { repo: "org/repo", ref: "abc123" }, ctx);
    expect(result.outcome).toBe("error");
    expect(result.message).toContain("403");
  });
});

describe("vcs.pr_status", () => {
  it("returns merged", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ merged: true, state: "closed", mergeable_state: "unknown" }),
    });

    const result = await prStatus("vcs.pr_status", { repo: "org/repo", pullNumber: 42 }, ctx);
    expect(result.outcome).toBe("merged");
  });

  it("returns mergeable", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ merged: false, state: "open", mergeable_state: "clean" }),
    });

    const result = await prStatus("vcs.pr_status", { repo: "org/repo", pullNumber: 42 }, ctx);
    expect(result.outcome).toBe("mergeable");
  });

  it("returns blocked", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ merged: false, state: "open", mergeable_state: "dirty" }),
    });

    const result = await prStatus("vcs.pr_status", { repo: "org/repo", pullNumber: 42 }, ctx);
    expect(result.outcome).toBe("blocked");
  });
});

describe("issue_tracker.comment_exists", () => {
  it("returns exists when pattern matches", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [
        { body: "Some text\n## Implementation Spec\ndetails here" },
      ],
    });

    const result = await commentExists(
      "issue_tracker.comment_exists",
      { repo: "org/repo", issueNumber: 10, pattern: "## Implementation Spec" },
      ctx,
    );
    expect(result.outcome).toBe("exists");
  });

  it("returns not_found when no match", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [{ body: "Just a regular comment" }],
    });

    const result = await commentExists(
      "issue_tracker.comment_exists",
      { repo: "org/repo", issueNumber: 10, pattern: "## Implementation Spec" },
      ctx,
    );
    expect(result.outcome).toBe("not_found");
  });
});

describe("vcs.pr_capacity", () => {
  it("returns available under limit", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [{}, {}],
    });

    const result = await prCapacity("vcs.pr_capacity", { repo: "org/repo", max: 4 }, ctx);
    expect(result.outcome).toBe("available");
    expect(result.message).toContain("2 open");
  });

  it("returns at_capacity at limit", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [{}, {}, {}, {}],
    });

    const result = await prCapacity("vcs.pr_capacity", { repo: "org/repo", max: 4 }, ctx);
    expect(result.outcome).toBe("at_capacity");
  });
});

describe("registerGitHubHandlers", () => {
  it("registers all 4 handlers", () => {
    const ops: string[] = [];
    registerGitHubHandlers((op) => ops.push(op));
    expect(ops.sort()).toEqual([
      "issue_tracker.comment_exists",
      "vcs.ci_status",
      "vcs.pr_capacity",
      "vcs.pr_status",
    ]);
  });
});

describe("token resolution", () => {
  it("throws when no token available", async () => {
    delete process.env.GH_TOKEN;
    delete process.env.GITHUB_TOKEN;

    await expect(ciStatus("vcs.ci_status", { repo: "org/repo", ref: "abc" }, ctx)).rejects.toThrow("No GitHub token");
  });

  it("falls back to GITHUB_TOKEN", async () => {
    delete process.env.GH_TOKEN;
    process.env.GITHUB_TOKEN = "fallback-token";

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ check_runs: [{ status: "completed", conclusion: "success" }] }),
    });

    const result = await ciStatus("vcs.ci_status", { repo: "org/repo", ref: "abc" }, ctx);
    expect(result.outcome).toBe("passed");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer fallback-token" }),
      }),
    );
  });
});

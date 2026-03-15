import { afterEach, describe, expect, it } from "vitest";
import {
  clearHandlers,
  evaluateGate,
  hasHandler,
  listHandlers,
  registerHandler,
  registerHandlers,
  type GateRequest,
} from "./gates.js";

afterEach(() => {
  clearHandlers();
});

describe("handler registry", () => {
  it("registers and lists handlers", () => {
    registerHandler("vcs.ci_status", async () => ({ outcome: "passing" }));
    registerHandler("issue_tracker.comment_exists", async () => ({ outcome: "found" }));

    expect(hasHandler("vcs.ci_status")).toBe(true);
    expect(hasHandler("unknown.op")).toBe(false);
    expect(listHandlers()).toEqual(["issue_tracker.comment_exists", "vcs.ci_status"]);
  });

  it("registers multiple handlers at once", () => {
    registerHandlers({
      "vcs.ci_status": async () => ({ outcome: "passing" }),
      "vcs.branch_exists": async () => ({ outcome: "exists" }),
    });

    expect(listHandlers()).toEqual(["vcs.branch_exists", "vcs.ci_status"]);
  });

  it("overwrites existing handler without throwing", () => {
    registerHandler("vcs.ci_status", async () => ({ outcome: "v1" }));
    registerHandler("vcs.ci_status", async () => ({ outcome: "v2" }));

    expect(listHandlers()).toEqual(["vcs.ci_status"]);
  });

  it("clears all handlers", () => {
    registerHandler("a.op", async () => ({ outcome: "ok" }));
    registerHandler("b.op", async () => ({ outcome: "ok" }));
    clearHandlers();

    expect(listHandlers()).toEqual([]);
  });
});

describe("evaluateGate", () => {
  const baseRequest: GateRequest = {
    gateId: "gate-1",
    entityId: "entity-1",
    op: "vcs.ci_status",
    params: { repo: "org/repo", pr: 42 },
  };

  it("returns outcome from handler", async () => {
    registerHandler("vcs.ci_status", async (_op, params) => ({
      outcome: "passing",
      message: `CI green for PR ${params.pr}`,
    }));

    const result = await evaluateGate(baseRequest);

    expect(result.gateId).toBe("gate-1");
    expect(result.entityId).toBe("entity-1");
    expect(result.op).toBe("vcs.ci_status");
    expect(result.outcome).toBe("passing");
    expect(result.message).toBe("CI green for PR 42");
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it("returns error outcome when no handler registered", async () => {
    const result = await evaluateGate({ ...baseRequest, op: "unknown.op" });

    expect(result.outcome).toBe("error");
    expect(result.message).toContain("No handler registered");
  });

  it("returns error outcome when handler throws", async () => {
    registerHandler("vcs.ci_status", async () => {
      throw new Error("GitHub API rate limited");
    });

    const result = await evaluateGate(baseRequest);

    expect(result.outcome).toBe("error");
    expect(result.message).toBe("GitHub API rate limited");
  });

  it("returns timeout outcome when handler exceeds timeout", async () => {
    registerHandler("vcs.ci_status", async () => {
      await new Promise((resolve) => setTimeout(resolve, 500));
      return { outcome: "passing" };
    });

    const result = await evaluateGate({ ...baseRequest, timeoutMs: 50 });

    expect(result.outcome).toBe("timeout");
    expect(result.message).toContain("timed out");
  });

  it("passes params and context to handler", async () => {
    let capturedOp: string | undefined;
    let capturedParams: Record<string, unknown> | undefined;
    let capturedContext: { entityId: string } | undefined;

    registerHandler("vcs.ci_status", async (op, params, context) => {
      capturedOp = op;
      capturedParams = params;
      capturedContext = context;
      return { outcome: "ok" };
    });

    await evaluateGate(baseRequest);

    expect(capturedOp).toBe("vcs.ci_status");
    expect(capturedParams).toEqual({ repo: "org/repo", pr: 42 });
    expect(capturedContext).toEqual({ entityId: "entity-1" });
  });

  it("defaults message to empty string when handler omits it", async () => {
    registerHandler("vcs.ci_status", async () => ({ outcome: "passing" }));

    const result = await evaluateGate(baseRequest);

    expect(result.message).toBe("");
  });

  it("uses default timeout when not specified", async () => {
    let ran = false;
    registerHandler("vcs.ci_status", async () => {
      ran = true;
      return { outcome: "ok" };
    });

    // No timeoutMs — should use default (300s), not fail
    await evaluateGate({ ...baseRequest, timeoutMs: undefined });

    expect(ran).toBe(true);
  });
});

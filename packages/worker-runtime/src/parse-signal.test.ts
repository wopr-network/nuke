import { describe, expect, it } from "vitest";
import { parseSignal } from "./parse-signal.js";

describe("parseSignal", () => {
  it("returns unknown for empty string", () => {
    expect(parseSignal("")).toEqual({ signal: "unknown", artifacts: {} });
  });

  it("returns unknown for unrecognised output", () => {
    expect(parseSignal("I did some stuff and it worked fine")).toEqual({ signal: "unknown", artifacts: {} });
  });

  it("parses spec_ready", () => {
    const { signal, artifacts } = parseSignal("Spec ready: WOP-1234");
    expect(signal).toBe("spec_ready");
    expect(artifacts).toEqual({ issueKey: "WOP-1234" });
  });

  it("parses pr_created with number", () => {
    const { signal, artifacts } = parseSignal("PR created: https://github.com/wopr-network/radar/pull/99");
    expect(signal).toBe("pr_created");
    expect(artifacts).toMatchObject({ prNumber: 99, prUrl: "https://github.com/wopr-network/radar/pull/99" });
  });

  it("parses clean", () => {
    const { signal, artifacts } = parseSignal("CLEAN: https://github.com/wopr-network/radar/pull/12");
    expect(signal).toBe("clean");
    expect(artifacts).toMatchObject({ url: "https://github.com/wopr-network/radar/pull/12" });
  });

  it("parses issues", () => {
    const { signal, artifacts } = parseSignal(
      "ISSUES: https://github.com/wopr-network/radar/pull/12 — missing types; unused import",
    );
    expect(signal).toBe("issues");
    expect(artifacts).toMatchObject({
      url: "https://github.com/wopr-network/radar/pull/12",
      reviewFindings: ["missing types", "unused import"],
    });
  });

  it("parses fixes_pushed", () => {
    const { signal, artifacts } = parseSignal("Fixes pushed: https://github.com/wopr-network/radar/pull/12");
    expect(signal).toBe("fixes_pushed");
    expect(artifacts).toMatchObject({ url: "https://github.com/wopr-network/radar/pull/12" });
  });

  it("parses merged", () => {
    const { signal, artifacts } = parseSignal("Merged: https://github.com/wopr-network/radar/pull/12");
    expect(signal).toBe("merged");
    expect(artifacts).toMatchObject({ url: "https://github.com/wopr-network/radar/pull/12" });
  });

  it("parses start", () => {
    expect(parseSignal("start")).toEqual({ signal: "start", artifacts: {} });
  });

  it("parses design_needed", () => {
    expect(parseSignal("design_needed")).toEqual({ signal: "design_needed", artifacts: {} });
  });

  it("parses design_ready", () => {
    expect(parseSignal("design_ready")).toEqual({ signal: "design_ready", artifacts: {} });
  });

  it("parses cant_resolve", () => {
    expect(parseSignal("cant_resolve")).toEqual({ signal: "cant_resolve", artifacts: {} });
  });

  it("picks signal from last matching line in multi-line output", () => {
    const output = [
      "I reviewed the code carefully.",
      "PR created: https://github.com/wopr-network/radar/pull/10",
      "Some trailing commentary.",
      "PR created: https://github.com/wopr-network/radar/pull/20",
    ].join("\n");
    const { signal, artifacts } = parseSignal(output);
    expect(signal).toBe("pr_created");
    expect(artifacts).toMatchObject({ prNumber: 20 });
  });

  it("ignores signal buried in the middle when later line matches too", () => {
    const output = ["spec_ready", "PR created: https://github.com/wopr-network/radar/pull/5"].join("\n");
    const { signal } = parseSignal(output);
    expect(signal).toBe("pr_created");
  });

  it("handles windows line endings", () => {
    expect(parseSignal("start\r")).toEqual({ signal: "start", artifacts: {} });
  });

  // wopr-changeset: documenting + learning
  it("parses docs_ready", () => {
    expect(parseSignal("Updated the README.\n\ndocs_ready")).toEqual({ signal: "docs_ready", artifacts: {} });
  });

  it("parses cant_document", () => {
    expect(parseSignal("cant_document")).toEqual({ signal: "cant_document", artifacts: {} });
  });

  it("parses learning_complete", () => {
    expect(parseSignal("learning_complete")).toEqual({ signal: "learning_complete", artifacts: {} });
  });

  it("parses cant_learn", () => {
    expect(parseSignal("cant_learn")).toEqual({ signal: "cant_learn", artifacts: {} });
  });

  it("does not match docs_ready mid-line", () => {
    expect(parseSignal("The docs_ready signal was emitted")).toEqual({ signal: "unknown", artifacts: {} });
  });

  // engineering flow bare-word signals
  it("parses bare spec_ready", () => {
    expect(parseSignal("I posted the spec.\n\nspec_ready")).toEqual({ signal: "spec_ready", artifacts: {} });
  });

  it("parses bare ci_failed", () => {
    expect(parseSignal("ci_failed")).toEqual({ signal: "ci_failed", artifacts: {} });
  });

  it("parses bare learned", () => {
    expect(parseSignal("Updated CLAUDE.md with findings.\n\nlearned")).toEqual({ signal: "learned", artifacts: {} });
  });

  it("parses bare blocked", () => {
    expect(parseSignal("blocked")).toEqual({ signal: "blocked", artifacts: {} });
  });

  it("parses bare closed", () => {
    expect(parseSignal("closed")).toEqual({ signal: "closed", artifacts: {} });
  });

  // wopr-incident signals
  it("parses triaged with severity", () => {
    const { signal, artifacts } = parseSignal("Triaged: WOP-500 severity=P1");
    expect(signal).toBe("triaged");
    expect(artifacts).toEqual({ issueKey: "WOP-500", severity: "P1" });
  });

  it("parses triaged P3", () => {
    const { signal, artifacts } = parseSignal("Triaged: WOP-501 severity=P3");
    expect(signal).toBe("triaged");
    expect(artifacts).toEqual({ issueKey: "WOP-501", severity: "P3" });
  });

  it("parses root_cause", () => {
    const { signal, artifacts } = parseSignal("Root cause: WOP-500 — null pointer in auth middleware");
    expect(signal).toBe("root_cause");
    expect(artifacts).toEqual({ issueKey: "WOP-500", rootCause: "null pointer in auth middleware" });
  });

  it("parses escalate", () => {
    const { signal, artifacts } = parseSignal("Escalate: WOP-500 — needs human expertise");
    expect(signal).toBe("escalate");
    expect(artifacts).toEqual({ issueKey: "WOP-500", reason: "needs human expertise" });
  });

  it("parses mitigated", () => {
    const { signal, artifacts } = parseSignal("Mitigated: WOP-500");
    expect(signal).toBe("mitigated");
    expect(artifacts).toEqual({ issueKey: "WOP-500" });
  });

  it("parses mitigation_failed", () => {
    const { signal, artifacts } = parseSignal("Mitigation failed: WOP-500 — rollback script not found");
    expect(signal).toBe("mitigation_failed");
    expect(artifacts).toEqual({ issueKey: "WOP-500", reason: "rollback script not found" });
  });

  it("parses resolved with PR URL", () => {
    const { signal, artifacts } = parseSignal("Resolved: WOP-500 — https://github.com/wopr-network/radar/pull/99");
    expect(signal).toBe("resolved");
    expect(artifacts).toEqual({ issueKey: "WOP-500", prUrl: "https://github.com/wopr-network/radar/pull/99" });
  });

  it("parses postmortem_complete", () => {
    const { signal, artifacts } = parseSignal("Postmortem complete: WOP-500");
    expect(signal).toBe("postmortem_complete");
    expect(artifacts).toEqual({ issueKey: "WOP-500" });
  });
});

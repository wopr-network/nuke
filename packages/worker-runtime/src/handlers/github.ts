// ─── GitHub Primitive Op Handlers ─────────────────────────────────────────────
//
// These run on the runner where GH_TOKEN and repo context are already available.
// No GitHub App token resolution needed — credentials injection handles that.

import { logger } from "../logger.js";
import type { PrimitiveHandler } from "../gates.js";

function getToken(): string {
  const token = process.env.GH_TOKEN ?? process.env.GITHUB_TOKEN;
  if (!token) throw new Error("No GitHub token available (GH_TOKEN or GITHUB_TOKEN)");
  return token;
}

const GITHUB_HEADERS = (token: string) => ({
  Authorization: `Bearer ${token}`,
  Accept: "application/vnd.github+json",
  "X-GitHub-Api-Version": "2022-11-28",
});

/** vcs.ci_status — check CI status on a commit ref */
export const ciStatus: PrimitiveHandler = async (_op, params, ctx) => {
  const token = getToken();
  const repo = params.repo as string;
  const ref = params.ref as string;
  if (!repo || !ref) return { outcome: "error", message: "Missing required params: repo, ref" };

  const res = await fetch(`https://api.github.com/repos/${repo}/commits/${ref}/check-runs?per_page=100`, {
    headers: GITHUB_HEADERS(token),
    signal: ctx.signal,
  });
  if (!res.ok) return { outcome: "error", message: `GitHub API error: ${res.status}` };

  const data = (await res.json()) as { check_runs: Array<{ conclusion: string | null; status: string }> };
  const runs = data.check_runs;

  if (runs.length === 0) return { outcome: "pending", message: "No check runs found" };
  const allComplete = runs.every((r) => r.status === "completed");
  if (!allComplete) return { outcome: "pending", message: "Check runs still in progress" };
  const allPassed = runs.every((r) => r.conclusion === "success" || r.conclusion === "skipped");
  return allPassed
    ? { outcome: "passed", message: "All checks passed" }
    : { outcome: "failed", message: "Some checks failed" };
};

/** vcs.pr_status — check PR merge status */
export const prStatus: PrimitiveHandler = async (_op, params, ctx) => {
  const token = getToken();
  const repo = params.repo as string;
  const pullNumber = Number(params.pullNumber);
  if (!repo || !pullNumber) return { outcome: "error", message: "Missing required params: repo, pullNumber" };

  const res = await fetch(`https://api.github.com/repos/${repo}/pulls/${pullNumber}`, {
    headers: GITHUB_HEADERS(token),
    signal: ctx.signal,
  });
  if (!res.ok) return { outcome: "error", message: `GitHub API error: ${res.status}` };

  const pr = (await res.json()) as { merged: boolean; state: string; mergeable_state: string };
  if (pr.merged) return { outcome: "merged" };
  if (pr.state === "closed") return { outcome: "closed" };
  if (pr.mergeable_state === "clean") return { outcome: "mergeable" };
  return { outcome: "blocked", message: `PR state: ${pr.mergeable_state}` };
};

/** issue_tracker.comment_exists — check if a comment matching a pattern exists */
export const commentExists: PrimitiveHandler = async (_op, params, ctx) => {
  const token = getToken();
  const repo = params.repo as string;
  const issueNumber = Number(params.issueNumber);
  const pattern = params.pattern as string;
  if (!repo || !issueNumber || !pattern) {
    return { outcome: "error", message: "Missing required params: repo, issueNumber, pattern" };
  }

  const res = await fetch(
    `https://api.github.com/repos/${repo}/issues/${issueNumber}/comments?per_page=100`,
    { headers: GITHUB_HEADERS(token), signal: ctx.signal },
  );
  if (!res.ok) return { outcome: "error", message: `GitHub API error: ${res.status}` };

  const comments = (await res.json()) as Array<{ body: string }>;
  // Use includes() instead of RegExp to avoid ReDoS from untrusted patterns
  const found = comments.some((c) => c.body.includes(pattern));
  return found
    ? { outcome: "exists", message: "Matching comment found" }
    : { outcome: "not_found", message: "No matching comment" };
};

/** vcs.pr_capacity — check if repo has capacity for more PRs */
export const prCapacity: PrimitiveHandler = async (_op, params, ctx) => {
  const token = getToken();
  const repo = params.repo as string;
  const max = Number(params.max ?? 4);
  if (!repo) return { outcome: "error", message: "Missing required param: repo" };

  const res = await fetch(`https://api.github.com/repos/${repo}/pulls?state=open&per_page=100`, {
    headers: GITHUB_HEADERS(token),
    signal: ctx.signal,
  });
  if (!res.ok) return { outcome: "error", message: `GitHub API error: ${res.status}` };

  const prs = (await res.json()) as Array<unknown>;
  return prs.length < max
    ? { outcome: "available", message: `${prs.length} open PRs (max ${max})` }
    : { outcome: "at_capacity", message: `${prs.length} open PRs (max ${max})` };
};

/** Register all GitHub handlers */
export function registerGitHubHandlers(register: (op: string, handler: PrimitiveHandler) => void): void {
  register("vcs.ci_status", ciStatus);
  register("vcs.pr_status", prStatus);
  register("vcs.pr_capacity", prCapacity);
  register("issue_tracker.comment_exists", commentExists);
  logger.info("[handlers/github] registered 4 handlers");
}

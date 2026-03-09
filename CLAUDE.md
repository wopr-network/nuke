# NUKE

Agent container runtime for the WOPR pipeline.

## Structure

- `packages/worker-runtime/` — HTTP server, SSE streaming, signal parsing
- `workers/coder/` — Dockerfile for engineering discipline (git, gh)
- `workers/devops/` — Dockerfile for devops discipline (git, curl)

## Check before committing

```bash
pnpm check
```

This runs biome lint + typecheck across all packages.

## Gotchas

- **Non-root user**: Containers run as `nuke` user, not root. Claude credentials go in `/home/nuke/.claude/`.
- **Signal parsing scans bottom-up**: `parseSignal()` reverses lines and returns first match. Last signal in output wins.
- **Session persistence**: `sessionId` from the `session` SSE event must be passed back on continue dispatches. `newSession: true` starts fresh.
- **LINEAR_API_KEY**: When set, the worker-runtime auto-configures a Linear MCP server via `mcp-remote`. No prompt configuration needed.
- **Port**: Defaults to `PORT=8080`. RADAR maps this to a dynamic host port via `docker run -p 0:8080`.
- **Body size limit**: `/dispatch` rejects request bodies over 1MB.
- **CLAUDECODE env var**: Explicitly deleted from the env passed to `query()` to prevent SDK conflicts.

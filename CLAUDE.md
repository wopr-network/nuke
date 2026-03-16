# HOLYSHIPPER

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

- **Non-root user**: Containers run as `holyshipper` user, not root. OpenCode config goes in `/home/holyshipper/.config/opencode/`.
- **OpenCode SDK**: Dispatches use `@opencode-ai/sdk` — NOT Claude Code. All inference goes through the holyship gateway (`HOLYSHIP_GATEWAY_URL`), metered via the credit ledger.
- **Gateway credentials**: `POST /credentials` accepts `{ gateway: { key: "sk-hs-..." }, gatewayUrl: "http://api:3001/v1", github: { token: "..." } }`.
- **Signal parsing scans bottom-up**: `parseSignal()` reverses lines and returns first match. Last signal in output wins.
- **Session persistence**: `sessionId` from the `session` SSE event must be passed back on continue dispatches. `newSession: true` starts fresh.
- **Port**: Defaults to `PORT=8080`. Holyship maps this to a dynamic host port via `docker run -p 0:8080`.
- **Body size limit**: `/dispatch` rejects request bodies over 1MB.

## Version Control: Prefer jj

Use `jj` (Jujutsu) for all VCS operations instead of `git`:
- `jj status`, `jj diff`, `jj log` for inspection
- `jj new` to start a change, `jj describe` to set the message
- `jj commit` to commit, `jj push` to push
- `jj squash`, `jj rebase`, `jj edit` for history manipulation

Fall back to `git` only for operations not yet supported by `jj`.

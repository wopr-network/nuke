# HOLYSHIPPER — Agent Containers

> The warheads in the WOPR stack. Docker containers that run Claude agents for a single invocation, stream results back to RADAR via SSE, and die.

---

## What This Is

HOLYSHIPPER is the agent container runtime for the [WOPR](https://github.com/wopr-network) agentic engineering pipeline. Each holyshipper is one agent invocation — an architect writing a spec, a coder implementing a feature, a reviewer reading a diff, a fixer addressing findings.

RADAR launches holyshippers. NORAD watches them. DEFCON decides if their output earns escalation. The holyshipper does the work.

**You fork this repo** to customize what's installed in your agent containers. A Python shop adds `pip`, `pytest`, `ruff`. A Rust shop adds `cargo`, `clippy`. The worker-runtime is shared — the tooling is yours.

---

## Architecture

```
packages/
  worker-runtime/       HTTP server + SSE streaming + signal parsing
    src/
      server.ts         POST /dispatch, GET /health
      types.ts          DispatchRequest, HolyshipperEvent (SSE event union)
      parse-signal.ts   Extract signal + artifacts from agent output
      index.ts          Public export surface (parseSignal, makeHandler, types)
      main.ts           Entrypoint (createServer, listen on PORT)

workers/
  coder/Dockerfile      node + git + gh (engineering discipline)
  devops/Dockerfile     node + git + curl (devops discipline)
```

### Worker Runtime

An HTTP server with two endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dispatch` | POST | Receive a prompt, run a Claude agent, stream SSE events back |
| `/health` | GET | Returns `{"ok": true}` |

### Per-Discipline Dockerfiles

Each discipline gets its own Dockerfile. The Dockerfile installs:

1. **System tools** — whatever the discipline needs (`git`, `gh`, `curl`, `pnpm`, etc.)
2. **Claude stack** — `@anthropic-ai/claude-code`, `@anthropic-ai/claude-agent-sdk`, `mcp-remote`
3. **Worker runtime** — copied from `packages/worker-runtime/`

The container runs as a non-root `holyshipper` user with `/workspace` as the working directory.

---

## Dispatch Protocol

RADAR POSTs to `/dispatch` with a JSON body:

```json
{
  "prompt": "You are a software engineer...",
  "modelTier": "sonnet",
  "newSession": true
}
```

For continue dispatches (multi-turn within the same entity):

```json
{
  "prompt": "Continue your work...",
  "modelTier": "sonnet",
  "sessionId": "ses_abc123"
}
```

### Model Tiers

| Tier | Model |
|------|-------|
| `opus` | claude-opus-4-6 |
| `sonnet` | claude-sonnet-4-6 |
| `haiku` | claude-haiku-4-5 |

### SSE Response

The holyshipper streams Server-Sent Events back to RADAR:

```
data: {"type":"session","sessionId":"abc-123"}

data: {"type":"tool_use","name":"Read","input":{"file_path":"/workspace/src/index.ts"}}

data: {"type":"text","text":"I'll implement the fix by..."}

data: {"type":"result","signal":"pr_created","artifacts":{"prUrl":"https://...","prNumber":456},"isError":false,"costUsd":0.042,"stopReason":"end_turn","subtype":"success"}
```

### Event Types

| Type | Fields | When |
|------|--------|------|
| `session` | `sessionId` | Once, at dispatch start |
| `system` | `subtype` | SDK system events |
| `tool_use` | `name`, `input` | Agent calls a tool (Read, Edit, Bash, etc.) |
| `text` | `text` | Agent produces text output |
| `result` | `signal`, `artifacts`, `isError`, `costUsd`, `stopReason` | Terminal event — dispatch complete |
| `error` | `message` | Agent crashed or timed out |

---

## Signal Parsing

When the agent finishes, the worker-runtime collects all text output and scans it **from the bottom up** for a recognized signal pattern. The signal and extracted artifacts are included in the `result` SSE event.

| Signal | Pattern | Artifacts |
|--------|---------|-----------|
| `spec_ready` | `Spec ready: WOP-123` | `{ issueKey }` |
| `pr_created` | `PR created: https://...pull/456` | `{ prUrl, prNumber }` |
| `clean` | `CLEAN: https://...` | `{ url }` |
| `issues` | `ISSUES: https://... — finding1; finding2` | `{ url, reviewFindings }` |
| `fixes_pushed` | `Fixes pushed: https://...` | `{ url }` |
| `merged` | `Merged: https://...` | `{ url }` |
| `cant_resolve` | `cant_resolve` | `{}` |
| `start` | `start` | `{}` |
| `design_needed` | `design_needed` | `{}` |
| `design_ready` | `design_ready` | `{}` |

Signals must appear on their own line. Last match wins. If no signal is recognized, the result contains `signal: "unknown"`.

---

## MCP Integration

When `LINEAR_API_KEY` is set in the container's environment, the worker-runtime automatically configures a Linear MCP server:

```
npx -y mcp-remote https://mcp.linear.app/mcp --header "Authorization: Bearer <key>"
```

This gives the agent access to Linear tools (read/write issues, comments, etc.) without any configuration in the prompt.

---

## Container Lifecycle

```
RADAR launches container (docker run)
  → container starts, HTTP server listens on PORT
  → RADAR POSTs to /dispatch
  → agent runs, streams SSE events
  → result event emitted with signal + artifacts
  → RADAR may POST again (continue dispatch, same sessionId)
  → ...
  → entity completes or times out
  → RADAR stops and removes the container
```

Containers are **stateful within an entity** — the session persists across continue dispatches. Containers are **ephemeral across entities** — each entity gets a fresh container.

---

## Adding a New Discipline

1. Create `workers/my-discipline/Dockerfile`
2. Start from `node:24-alpine`
3. Install your discipline's system tools (`apk add --no-cache ...`)
4. Install the Claude stack:
   ```dockerfile
   RUN npm install -g @anthropic-ai/claude-code @anthropic-ai/claude-agent-sdk mcp-remote
   ```
5. Copy and build the worker runtime:
   ```dockerfile
   COPY packages/worker-runtime /app/worker-runtime
   WORKDIR /app/worker-runtime
   RUN npm install --production
   ```
6. Set up the non-root user, secrets mount, and entrypoint (see existing Dockerfiles)
7. Build: `docker build -f workers/my-discipline/Dockerfile -t holyshipper-my-discipline .`

---

## Development

```bash
# Install dependencies
pnpm install

# Build
pnpm build

# Run tests
pnpm test

# Lint + format
pnpm lint
pnpm format

# Full check (lint + typecheck)
pnpm check
```

---

## The Stack

HOLYSHIPPER is one piece of the WOPR agentic engineering pipeline:

- [WOPR](https://github.com/wopr-network/wopr) — the AI
- [DEFCON](https://github.com/wopr-network/defcon) — the state machine engine
- [RADAR](https://github.com/wopr-network/radar) — detection and dispatch
- **HOLYSHIPPER** (this repo) — agent containers
- [NORAD](https://github.com/wopr-network/norad) — the command center dashboard
- [Bunker](https://github.com/wopr-network/bunker) — flow definitions and reference implementation

See [The Thesis](https://github.com/wopr-network/defcon/blob/main/docs/method/manifesto/the-thesis.md) for why this exists.

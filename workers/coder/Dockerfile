FROM node:24-alpine

# System tools the coder discipline needs
RUN apk add --no-cache git github-cli

# Claude Code CLI + Agent SDK + MCP bridge
RUN npm install -g \
  @anthropic-ai/claude-code \
  @anthropic-ai/claude-agent-sdk \
  mcp-remote

# Worker runtime
COPY packages/worker-runtime /app/worker-runtime
WORKDIR /app/worker-runtime
RUN npm install --production

# Secrets mount point (populated by radar at runtime)
RUN mkdir -p /run/secrets

# Non-root user
RUN addgroup -S nuke && adduser -S nuke -G nuke
RUN mkdir -p /home/nuke/.claude /workspace && chown -R nuke:nuke /home/nuke /workspace
USER nuke

WORKDIR /workspace

ENV PORT=8080
EXPOSE 8080

CMD ["node", "/app/worker-runtime/dist/main.js"]

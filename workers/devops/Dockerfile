FROM node:24-alpine

# System tools the devops discipline needs
RUN apk add --no-cache git curl

# Claude Code CLI + Agent SDK + MCP bridge
RUN npm install -g \
  @anthropic-ai/claude-code \
  @anthropic-ai/claude-agent-sdk \
  mcp-remote

# Worker runtime
COPY packages/worker-runtime /app/worker-runtime
WORKDIR /app/worker-runtime
RUN npm install --production

RUN mkdir -p /run/secrets

RUN addgroup -S holyshipper && adduser -S holyshipper -G holyshipper
RUN mkdir -p /home/holyshipper/.claude /workspace && chown -R holyshipper:holyshipper /home/holyshipper /workspace
USER holyshipper

WORKDIR /workspace

ENV PORT=8080
EXPOSE 8080

CMD ["node", "/app/worker-runtime/dist/main.js"]

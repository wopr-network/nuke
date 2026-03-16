FROM node:24-alpine

# System tools the devops discipline needs
RUN apk add --no-cache git curl

# OpenCode CLI — open source coding agent (replaces Claude Code)
RUN npm install -g opencode

# Worker runtime
COPY packages/worker-runtime /app/worker-runtime
WORKDIR /app/worker-runtime
RUN npm install --production

RUN mkdir -p /run/secrets

RUN addgroup -S holyshipper && adduser -S holyshipper -G holyshipper
RUN mkdir -p /home/holyshipper/.config/opencode /workspace && chown -R holyshipper:holyshipper /home/holyshipper /workspace
USER holyshipper

WORKDIR /workspace

ENV PORT=8080
EXPOSE 8080

CMD ["node", "/app/worker-runtime/dist/main.js"]

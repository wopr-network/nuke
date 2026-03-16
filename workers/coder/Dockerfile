FROM node:24-slim

# System tools the coder discipline needs
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  bash \
  build-essential \
  python3 \
  curl \
  ca-certificates \
  && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
     -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
     > /etc/apt/sources.list.d/github-cli.list \
  && apt-get update && apt-get install -y --no-install-recommends gh \
  && rm -rf /var/lib/apt/lists/*

# OpenCode CLI — open source coding agent (replaces Claude Code)
RUN npm install -g opencode

# Worker runtime — copy package.json + dist only (no pnpm artifacts)
COPY packages/worker-runtime/package.json /app/worker-runtime/
WORKDIR /app/worker-runtime
RUN npm install --production
COPY packages/worker-runtime/dist /app/worker-runtime/dist

# Secrets mount point (populated at runtime)
RUN mkdir -p /run/secrets

# Non-root user
RUN groupadd -r holyshipper && useradd -r -g holyshipper -m holyshipper
RUN mkdir -p /home/holyshipper/.config/opencode /workspace && chown -R holyshipper:holyshipper /home/holyshipper /workspace
USER holyshipper

WORKDIR /workspace

ENV PORT=8080
ENV LOG_LEVEL=debug
EXPOSE 8080

CMD ["node", "/app/worker-runtime/dist/main.js"]

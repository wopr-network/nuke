# Build stage — compile worker-runtime TypeScript
FROM node:24-slim AS build
RUN npm install -g pnpm
WORKDIR /build
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY packages/worker-runtime/package.json packages/worker-runtime/
COPY packages/worker-runtime/tsconfig.json packages/worker-runtime/
COPY packages/worker-runtime/src packages/worker-runtime/src
RUN pnpm install --frozen-lockfile
RUN pnpm -C packages/worker-runtime run build

# Runtime stage
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

# Worker runtime — copy package.json + built dist from build stage
COPY --from=build /build/packages/worker-runtime/package.json /app/worker-runtime/
WORKDIR /app/worker-runtime
RUN npm install --production
COPY --from=build /build/packages/worker-runtime/dist /app/worker-runtime/dist

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

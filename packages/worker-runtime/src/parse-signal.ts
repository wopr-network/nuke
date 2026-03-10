interface SignalPattern {
  pattern: RegExp;
  signal: string;
  extractArtifacts: (match: RegExpMatchArray) => Record<string, unknown>;
}

const SIGNAL_PATTERNS: SignalPattern[] = [
  {
    pattern: /Spec ready:\s*(WOP-\d+)/,
    signal: "spec_ready",
    extractArtifacts: (m) => ({ issueKey: m[1] }),
  },
  {
    pattern: /PR created:\s*(https:\/\/github\.com\/[^\s]+\/pull\/(\d+))/,
    signal: "pr_created",
    extractArtifacts: (m) => ({ prUrl: m[1], prNumber: Number(m[2]) }),
  },
  {
    pattern: /CLEAN:\s*(https:\/\/[^\s]+)/,
    signal: "clean",
    extractArtifacts: (m) => ({ url: m[1] }),
  },
  {
    pattern: /ISSUES:\s*(https:\/\/[^\s]+)\s*[—–-]\s*(.+)/,
    signal: "issues",
    extractArtifacts: (m) => ({
      url: m[1],
      reviewFindings: m[2]
        .split(";")
        .map((s) => s.trim())
        .filter(Boolean),
    }),
  },
  {
    pattern: /Fixes pushed:\s*(https:\/\/[^\s]+)/,
    signal: "fixes_pushed",
    extractArtifacts: (m) => ({ url: m[1] }),
  },
  {
    pattern: /Merged:\s*(https:\/\/[^\s]+)/,
    signal: "merged",
    extractArtifacts: (m) => ({ url: m[1] }),
  },
  {
    pattern: /^start\r?$/,
    signal: "start",
    extractArtifacts: () => ({}),
  },
  {
    pattern: /^design_needed\r?$/,
    signal: "design_needed",
    extractArtifacts: () => ({}),
  },
  {
    pattern: /^design_ready\r?$/,
    signal: "design_ready",
    extractArtifacts: () => ({}),
  },
  {
    pattern: /^cant_resolve\r?$/,
    signal: "cant_resolve",
    extractArtifacts: () => ({}),
  },
  // wopr-changeset: documenting + learning bare-word signals
  {
    pattern: /^docs_ready\r?$/,
    signal: "docs_ready",
    extractArtifacts: () => ({}),
  },
  {
    pattern: /^cant_document\r?$/,
    signal: "cant_document",
    extractArtifacts: () => ({}),
  },
  {
    pattern: /^learning_complete\r?$/,
    signal: "learning_complete",
    extractArtifacts: () => ({}),
  },
  {
    pattern: /^cant_learn\r?$/,
    signal: "cant_learn",
    extractArtifacts: () => ({}),
  },
  // wopr-incident: structured signals with artifact extraction
  {
    pattern: /Triaged:\s*(\S+)\s+severity=(P[123])/,
    signal: "triaged",
    extractArtifacts: (m) => ({ issueKey: m[1], severity: m[2] }),
  },
  {
    pattern: /Root cause:\s*(\S+)\s*[—–-]\s*(.+)/,
    signal: "root_cause",
    extractArtifacts: (m) => ({ issueKey: m[1], rootCause: m[2].trim() }),
  },
  {
    pattern: /Escalate:\s*(\S+)\s*[—–-]\s*(.+)/,
    signal: "escalate",
    extractArtifacts: (m) => ({ issueKey: m[1], reason: m[2].trim() }),
  },
  {
    pattern: /Mitigated:\s*(\S+)/,
    signal: "mitigated",
    extractArtifacts: (m) => ({ issueKey: m[1] }),
  },
  {
    pattern: /Mitigation failed:\s*(\S+)\s*[—–-]\s*(.+)/,
    signal: "mitigation_failed",
    extractArtifacts: (m) => ({ issueKey: m[1], reason: m[2].trim() }),
  },
  {
    pattern: /Resolved:\s*(\S+)\s*[—–-]\s*(https:\/\/[^\s]+)/,
    signal: "resolved",
    extractArtifacts: (m) => ({ issueKey: m[1], prUrl: m[2] }),
  },
  {
    pattern: /Postmortem complete:\s*(\S+)/,
    signal: "postmortem_complete",
    extractArtifacts: (m) => ({ issueKey: m[1] }),
  },
];

export function parseSignal(output: string): {
  signal: string;
  artifacts: Record<string, unknown>;
} {
  const lines = output.split("\n").reverse();
  for (const line of lines) {
    for (const { pattern, signal, extractArtifacts } of SIGNAL_PATTERNS) {
      const match = line.match(pattern);
      if (match) {
        return { signal, artifacts: extractArtifacts(match) };
      }
    }
  }
  return { signal: "unknown", artifacts: {} };
}

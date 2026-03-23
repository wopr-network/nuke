// ─── Inbound ─────────────────────────────────────────────────────────────────

export interface DispatchRequest {
  prompt: string;
  modelTier: "opus" | "sonnet" | "haiku" | "deepseek" | "test";
  /** Omit or set false to start a fresh session */
  sessionId?: string | null;
  newSession?: boolean;
}

// ─── Outbound SSE events ──────────────────────────────────────────────────────

export interface SessionEvent {
  type: "session";
  sessionId: string;
}

export interface SystemEvent {
  type: "system";
  subtype: string;
}

export interface ToolUseEvent {
  type: "tool_use";
  name: string;
  input: Record<string, unknown>;
}

export interface TextEvent {
  type: "text";
  text: string;
}

export interface ResultEvent {
  type: "result";
  subtype: string;
  isError: boolean;
  stopReason: string | null;
  costUsd: number | null;
  signal: string;
  artifacts: Record<string, unknown>;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type HolyshipperEvent = SessionEvent | SystemEvent | ToolUseEvent | TextEvent | ResultEvent | ErrorEvent;

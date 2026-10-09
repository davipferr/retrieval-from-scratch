// Types shared by the Node server and the React UI.

export type Mode = "hybrid" | "bm25" | "vector";
export type ModelId = "claude-sonnet-5-5" | "claude-opus-5-5";
export type Effort = "low" | "medium" | "high";

export const MODELS: { id: ModelId; label: string; hint: string }[] = [
  { id: "claude-sonnet-5-5", label: "Sonnet 5.5", hint: "fast, cheap" },
  { id: "claude-opus-5-5", label: "Opus 5.5", hint: "best answers" },
];

/** One retrieved chunk, as returned by the Rust `hybrid` server. */
export interface Hit {
  id: number;
  path: string;
  abs_path: string;
  start_line: number;
  end_line: number;
  heading: string | null;
  text: string;
  /** RRF score in hybrid mode, otherwise the engine's own score. */
  score: number;
  bm25_rank: number | null;
  vector_rank: number | null;
  bm25_score: number | null;
  similarity: number | null;
}

export interface Timings {
  bm25_ms: number;
  embed_ms: number;
  vector_ms: number;
  total_ms: number;
}

export interface SearchResponse {
  query: string;
  mode: Mode;
  hits: Hit[];
  timings: Timings;
}

export interface IndexStats {
  chunks: number;
  files: number;
  dim: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  messages: ChatTurn[];
  model: ModelId;
  mode: Mode;
  k: number;
  effort: Effort;
}

/** Server-sent events streamed by POST /api/chat, in order. */
export type ChatEvent =
  | { type: "sources"; query: string; hits: Hit[]; timings: Timings }
  | { type: "thinking"; text: string }
  | { type: "text"; text: string }
  | { type: "fallback"; from: string; to: string }
  | {
      type: "done";
      model: string;
      stop_reason: string | null;
      refusal_category: string | null;
      usage: { input_tokens: number; output_tokens: number };
    }
  | { type: "error"; message: string };

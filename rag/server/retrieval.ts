// Client for the Rust `hybrid` server (crates/hybrid), which owns chunking,
// embeddings, BM25, HNSW and Reciprocal Rank Fusion. This side only asks for
// the top-k chunks.

import type { ChatTurn, IndexStats, Mode, SearchResponse } from "../shared/types.ts";

export const RUST_URL = process.env.RUST_URL ?? "http://127.0.0.1:4000";

export class RetrievalError extends Error {}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${RUST_URL}${path}`, { signal });
  } catch (err) {
    if (signal?.aborted) throw err;
    throw new RetrievalError(
      `Can't reach the retrieval server at ${RUST_URL}. Start it with: cargo run --release -p hybrid -- serve`,
    );
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { error?: string } | null;
    throw new RetrievalError(body?.error ?? `retrieval server returned ${res.status}`);
  }
  return (await res.json()) as T;
}

export function search(query: string, mode: Mode, k: number, signal?: AbortSignal) {
  const params = new URLSearchParams({ q: query, mode, k: String(k) });
  return get<SearchResponse>(`/api/search?${params}`, signal);
}

export function stats() {
  return get<IndexStats>("/api/stats");
}

/**
 * What to search for, given the conversation so far.
 *
 * Usually the latest question. But a short follow-up ("and the tests?")
 * means nothing on its own, so it borrows the previous question's words.
 * (A sturdier approach is to have a small model rewrite the follow-up into a
 * standalone query — at the cost of an extra API call per turn.)
 */
export function retrievalQuery(messages: ChatTurn[]): string {
  const users = messages.filter((m) => m.role === "user").map((m) => m.content.trim());
  const last = users.at(-1) ?? "";
  const previous = users.at(-2);
  if (previous && last.split(/\s+/).length < 5) return `${previous} ${last}`;
  return last;
}

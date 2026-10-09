import type { ChatEvent, ChatRequest, IndexStats } from "../../shared/types.ts";

export async function health(): Promise<{ retrieval: IndexStats | null; error?: string }> {
  const res = await fetch("/api/health");
  return res.json();
}

/**
 * POSTs a chat request and calls `onEvent` for each server-sent event.
 * (`EventSource` only does GET, so the SSE stream is parsed by hand: events
 * are separated by a blank line, payload lines start with `data:`.)
 */
export async function streamChat(
  req: ChatRequest,
  onEvent: (event: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.error ?? `server returned ${res.status}`);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data) as ChatEvent);
    }
  }
}

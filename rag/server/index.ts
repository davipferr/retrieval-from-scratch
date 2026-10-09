// HTTP server for the RAG app.
//
//   GET  /api/health   is retrieval up? how big is the index?
//   GET  /api/search   retrieval only (proxied to the Rust server)
//   POST /api/chat     retrieval + Claude, streamed as server-sent events
//
// In production (`npm start` after `npm run build`) it also serves the UI.

import { existsSync } from "node:fs";

import Anthropic from "@anthropic-ai/sdk";
import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";

import { MODELS, type ChatEvent, type ChatRequest, type Mode } from "../shared/types.ts";
import { chat } from "./chat.ts";
import { RetrievalError, RUST_URL, search, stats } from "./retrieval.ts";

const PORT = Number(process.env.RAG_PORT ?? 3001);
const MODES: Mode[] = ["hybrid", "bm25", "vector"];
const EFFORTS = ["low", "medium", "high"];

const app = new Hono();

app.get("/api/health", async (c) => {
  try {
    return c.json({ retrieval: await stats() });
  } catch (err) {
    return c.json({ retrieval: null, error: (err as Error).message });
  }
});

app.get("/api/search", async (c) => {
  const q = c.req.query("q")?.trim();
  const mode = (c.req.query("mode") ?? "hybrid") as Mode;
  if (!q || !MODES.includes(mode)) return c.json({ error: "need q and a valid mode" }, 400);
  try {
    return c.json(await search(q, mode, Number(c.req.query("k") ?? 8)));
  } catch (err) {
    return c.json({ error: (err as Error).message }, 502);
  }
});

function parseChatRequest(body: unknown): ChatRequest | string {
  const b = body as Partial<ChatRequest> | null;
  if (!b || !Array.isArray(b.messages) || b.messages.length === 0) return "messages must be a non-empty array";
  const valid = b.messages.every(
    (m) => (m.role === "user" || m.role === "assistant") && typeof m.content === "string" && m.content.trim(),
  );
  if (!valid) return "every message needs a role (user/assistant) and non-empty content";
  if (b.messages.at(-1)!.role !== "user") return "the last message must be from the user";
  if (!MODELS.some((m) => m.id === b.model)) return `model must be one of ${MODELS.map((m) => m.id).join(", ")}`;
  return {
    messages: b.messages,
    model: b.model!,
    mode: MODES.includes(b.mode as Mode) ? b.mode! : "hybrid",
    k: Math.min(Math.max(Number(b.k) || 8, 1), 20),
    effort: EFFORTS.includes(b.effort as string) ? b.effort! : "medium",
  };
}

/** Turns failures into a message the UI can show. Most specific first. */
function describeError(err: unknown): string {
  if (err instanceof RetrievalError) return err.message;
  if (err instanceof Anthropic.AuthenticationError)
    return "Claude rejected the credentials. Set ANTHROPIC_API_KEY in rag/.env (see .env.example).";
  if (err instanceof Anthropic.RateLimitError) return "Rate limited by the Claude API; try again in a moment.";
  if (err instanceof Anthropic.APIConnectionError) {
    // e.g. SELF_SIGNED_CERT_IN_CHAIN when antivirus/proxy HTTPS inspection
    // re-signs traffic; `--use-system-ca` (in package.json) handles that case.
    const cause = err.cause as { cause?: { code?: string }; code?: string } | undefined;
    const code = cause?.cause?.code ?? cause?.code;
    return `Can't connect to the Claude API${code ? ` (${code})` : ""}. Check your network or proxy.`;
  }
  if (err instanceof Anthropic.APIError) return `Claude API error ${err.status ?? ""}: ${err.message}`;
  // A client-side error before any request, typically no credentials at all.
  if (err instanceof Anthropic.AnthropicError)
    return `${err.message}\n\nTo fix: copy rag/.env.example to rag/.env and set ANTHROPIC_API_KEY (restart the server after).`;
  return err instanceof Error ? err.message : String(err);
}

app.post("/api/chat", async (c) => {
  const req = parseChatRequest(await c.req.json().catch(() => null));
  if (typeof req === "string") return c.json({ error: req }, 400);

  return streamSSE(c, async (sse) => {
    const abort = new AbortController();
    sse.onAbort(() => abort.abort()); // browser closed or hit stop
    const send = (event: ChatEvent) => sse.writeSSE({ data: JSON.stringify(event) });
    try {
      for await (const event of chat(req, abort.signal)) await send(event);
    } catch (err) {
      if (!abort.signal.aborted) {
        console.error(err);
        await send({ type: "error", message: describeError(err) });
      }
    }
  });
});

if (process.argv.includes("--prod")) {
  if (!existsSync("dist/index.html")) throw new Error("run `npm run build` first");
  app.use("/*", serveStatic({ root: "./dist" }));
}

serve({ fetch: app.fetch, port: PORT, hostname: "127.0.0.1" }, ({ port }) => {
  console.log(`RAG server on http://127.0.0.1:${port} (retrieval: ${RUST_URL})`);
});

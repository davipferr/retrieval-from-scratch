import { useEffect, useRef, useState } from "react";

import { MODELS, type ChatEvent, type ChatTurn, type Effort, type IndexStats, type Mode, type ModelId } from "../../shared/types.ts";
import { Answer } from "./Answer.tsx";
import { health, streamChat } from "./api.ts";
import { Sources, type Retrieved } from "./Sources.tsx";

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  // assistant only
  status?: "retrieving" | "thinking" | "streaming" | "done" | "error" | "stopped";
  thinking?: string;
  retrieved?: Retrieved;
  mode?: Mode;
  fallback?: { from: string; to: string };
  error?: string;
  meta?: { model: string; input: number; output: number; refusal: string | null };
}

const SUGGESTIONS = [
  "How does the HNSW index decide which neighbors to keep?",
  "Where is FlowNodeView used?",
  "How are posting lists compressed?",
];

let nextId = 1;

function load<T extends string>(key: string, fallback: T, allowed: readonly string[]): T {
  try {
    const v = localStorage.getItem(key);
    return v && allowed.includes(v) ? (v as T) : fallback;
  } catch {
    return fallback;
  }
}

function save(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* private mode etc. */
  }
}

export function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [model, setModel] = useState<ModelId>(() => load("model", "claude-sonnet-5-5", MODELS.map((m) => m.id)));
  const [mode, setMode] = useState<Mode>(() => load("mode", "hybrid", ["hybrid", "bm25", "vector"]));
  const [effort, setEffort] = useState<Effort>(() => load("effort", "medium", ["low", "medium", "high"]));
  const [stats, setStats] = useState<IndexStats | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [highlight, setHighlight] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const busy = messages.at(-1)?.status !== undefined && ["retrieving", "thinking", "streaming"].includes(messages.at(-1)!.status!);

  useEffect(() => {
    health()
      .then((h) => {
        setStats(h.retrieval);
        setBackendError(h.error ?? null);
      })
      .catch(() => setBackendError("Can't reach the Node server."));
  }, []);

  useEffect(() => {
    save("model", model);
    save("mode", mode);
    save("effort", effort);
  }, [model, mode, effort]);
  useEffect(() => {
    // (block body: scrollIntoView returns a Promise in newer browsers, and an
    // effect may only return a cleanup function)
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages.length]);

  const update = (id: number, fn: (m: Message) => Message) =>
    setMessages((ms) => ms.map((m) => (m.id === id ? fn(m) : m)));

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    const user: Message = { id: nextId++, role: "user", content: question };
    const reply: Message = { id: nextId++, role: "assistant", content: "", status: "retrieving", mode };
    // Only completed text goes back to the server as history.
    const history: ChatTurn[] = [...messages, user]
      .filter((m) => m.role === "user" || (m.status === "done" && m.content))
      .map((m) => ({ role: m.role, content: m.content }));
    setMessages((ms) => [...ms, user, reply]);
    setSelected(reply.id);
    setHighlight(null);
    setInput("");

    const abort = new AbortController();
    abortRef.current = abort;
    const onEvent = (e: ChatEvent) =>
      update(reply.id, (m) => {
        switch (e.type) {
          case "sources":
            return { ...m, status: "thinking", retrieved: { query: e.query, hits: e.hits, timings: e.timings } };
          case "thinking":
            return { ...m, thinking: (m.thinking ?? "") + e.text };
          case "text":
            return { ...m, status: "streaming", content: m.content + e.text };
          case "fallback":
            return { ...m, fallback: { from: e.from, to: e.to } };
          case "done":
            return {
              ...m,
              status: "done",
              meta: { model: e.model, input: e.usage.input_tokens, output: e.usage.output_tokens, refusal: e.refusal_category },
              error: e.stop_reason === "refusal" ? "The request was declined by the model's safety checks." : m.error,
            };
          case "error":
            return { ...m, status: "error", error: e.message };
        }
      });
    try {
      await streamChat({ messages: history, model, mode, k: 8, effort }, onEvent, abort.signal);
    } catch (err) {
      update(reply.id, (m) =>
        abort.signal.aborted ? { ...m, status: "stopped" } : { ...m, status: "error", error: (err as Error).message },
      );
    } finally {
      // a stream that ended without a "done" event was cut off
      update(reply.id, (m) => (m.status === "streaming" || m.status === "thinking" ? { ...m, status: "stopped" } : m));
      abortRef.current = null;
    }
  }

  const selectedMessage = messages.find((m) => m.id === selected);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Repo RAG</h1>
          <span className="muted small">
            {stats ? `${stats.chunks.toLocaleString()} chunks · ${stats.files} files · ${stats.dim}-d` : "retrieval offline"}
          </span>
        </div>
        <div className="controls">
          <label>
            Model
            <select value={model} onChange={(e) => setModel(e.target.value as ModelId)}>
              {MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label} — {m.hint}
                </option>
              ))}
            </select>
          </label>
          <label>
            Retrieval
            <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              <option value="hybrid">Hybrid (RRF)</option>
              <option value="bm25">BM25 only</option>
              <option value="vector">Vector only</option>
            </select>
          </label>
          <label>
            Effort
            <select value={effort} onChange={(e) => setEffort(e.target.value as Effort)}>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </label>
        </div>
      </header>

      {backendError && <div className="banner">{backendError}</div>}

      <main className="layout">
        <section className="chat">
          <div className="messages">
            {messages.length === 0 && (
              <div className="empty">
                <p>Ask about your code and notes. Answers cite the chunks they came from.</p>
                <div className="suggestions">
                  {SUGGESTIONS.map((s) => (
                    <button key={s} type="button" onClick={() => send(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m) =>
              m.role === "user" ? (
                <div key={m.id} className="msg msg-user">
                  {m.content}
                </div>
              ) : (
                <div
                  key={m.id}
                  className={`msg msg-assistant ${selected === m.id ? "msg-selected" : ""}`}
                  onClick={() => setSelected(m.id)}
                >
                  {m.status === "retrieving" && <p className="status">Searching your repos…</p>}
                  {m.thinking && (
                    <details className="thinking" open={m.status === "thinking"}>
                      <summary>{m.status === "thinking" ? "Thinking…" : "Thought process"}</summary>
                      <p>{m.thinking}</p>
                    </details>
                  )}
                  {m.status === "thinking" && !m.thinking && <p className="status">Thinking…</p>}
                  {m.fallback && (
                    <p className="note">
                      {m.fallback.from} declined; {m.fallback.to} answered instead.
                    </p>
                  )}
                  {m.content && (
                    <Answer
                      text={m.content}
                      onCite={(n) => {
                        setSelected(m.id);
                        setHighlight(n);
                      }}
                    />
                  )}
                  {m.error && <p className="error">{m.error}</p>}
                  {m.status === "stopped" && <p className="note">Stopped.</p>}
                  {m.meta && (
                    <p className="meta">
                      {m.meta.model} · {m.meta.input.toLocaleString()} in / {m.meta.output.toLocaleString()} out tokens ·{" "}
                      {m.retrieved?.hits.length ?? 0} sources ({m.mode})
                    </p>
                  )}
                </div>
              ),
            )}
            <div ref={bottomRef} />
          </div>

          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
          >
            <textarea
              value={input}
              placeholder="Ask about your repos…"
              rows={2}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(input);
                }
              }}
            />
            {busy ? (
              <button type="button" className="secondary" onClick={() => abortRef.current?.abort()}>
                Stop
              </button>
            ) : (
              <button type="submit" disabled={!input.trim()}>
                Send
              </button>
            )}
          </form>
        </section>

        <Sources retrieved={selectedMessage?.retrieved} mode={selectedMessage?.mode ?? mode} highlight={highlight} />
      </main>
    </div>
  );
}

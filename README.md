# Search & RAG from scratch

Two learning projects that work together: a search engine and a vector database
written in Rust, feeding a RAG chat app in TypeScript that answers questions
about your own repos and notes with Claude.

```
React UI (rag/web) ──► Node + Claude SDK (rag/server)      RAG: retrieve → prompt → stream
                              │ HTTP :4000
                              ▼
          Rust workspace
          ├── crates/search-engine   crawler, inverted index, BM25, typo tolerance   (Project 1)
          ├── crates/vector-db       SIMD distances, HNSW, mmap, local embeddings     (Project 2)
          └── crates/hybrid          chunking + both engines + Reciprocal Rank Fusion, HTTP API
```

**Why hybrid?** Keyword search finds exact names (`FlowNodeView`, an error code)
that embeddings blur. Embeddings find meaning ("how is the graph saved to disk?"
→ `persist.rs`) that keywords miss. Both run on every query, and RRF merges the
two ranked lists.

## Quick start

```powershell
# 1. Models (once): ONNX Runtime DLL + bge-small-en-v1.5 into models/
powershell -ExecutionPolicy Bypass -File scripts/fetch-models.ps1

# 2. Index the folders you want to ask about
cargo run --release -p hybrid -- ingest C:\Users\me\Documents\projetos\react crates

# 3. Retrieval server on :4000
cargo run --release -p hybrid -- serve

# 4. Chat app on http://localhost:5173 (in another terminal)
cd rag; copy .env.example .env    # then set ANTHROPIC_API_KEY in .env
npm install; npm run dev
```

Production build of the UI: `npm run build && npm start` (serves on :3001).

## The crates

### `search-engine`: Project 1
Tokenizer (accent folding, Porter stemmer), delta + varint posting lists,
mmapped inverted index, BM25, and typo tolerance via a Levenshtein automaton.
See [crates/search-engine/README.md](crates/search-engine/README.md).

### `vector-db`: Project 2, Rust half

| Part | File | What to look at |
|---|---|---|
| Flat storage + brute force | `vectors.rs`, `flat.rs` | contiguous `f32`s; top-k with a size-k min-heap |
| SIMD | `distance.rs` | scalar vs `wide::f32x8` vs AVX2+FMA intrinsics, picked at runtime |
| HNSW | `hnsw.rs` | layered graph, neighbor-selection heuristic, recall vs brute force |
| Persistence | `persist.rs` | vectors and the CSR graph cast straight out of `mmap` |
| Embeddings | `embed/wordpiece.rs`, `embed/onnx.rs` | hand-written BERT tokenizer; ONNX Runtime via `ort` |

```powershell
cargo run --release -p vector-db -- bench          # SIMD speedup, HNSW recall/speed vs ef
cargo run --release -p vector-db -- embed "free memory in Rust" "the Drop trait" "pizza"
cargo run --release -p vector-db -- tokenize "HNSW tokenization"
cargo run --release -p vector-db -- recall data/vectors
```

Measured on this machine (i7-13700H, 20k × 384-d):

| | recall@10 | ms/query |
|---|---|---|
| brute force | 1.000 | 0.89 |
| HNSW ef=64 | 0.974 | 0.09 (10×) |
| HNSW ef=128 | 0.998 | 0.14 (7×) |

Dot product over 20k vectors: scalar 3.3 ms, `wide` 1.3 ms, AVX2+FMA 0.95 ms (3.5×).

### `hybrid`: the glue

- `chunk.rs`: line-based chunking that splits on blank lines and Markdown headings and packs blocks to ~1,000 chars. Every chunk maps back to `path:start-end`.
- `retriever.rs`: ingest (one id per chunk in *both* engines), search in three modes.
- `rrf.rs`: Reciprocal Rank Fusion, `Σ 1/(60 + rank)`.

```powershell
cargo run --release -p hybrid -- compare "FlowNodeView"     # BM25 vs vector vs hybrid
cargo run --release -p hybrid -- search "how are posting lists compressed" --mode vector
```

API: `GET /api/search?q=…&k=8&mode=hybrid|bm25|vector`, `GET /api/stats`.

## `rag/`: Project 2, TypeScript half

- `server/chat.ts`: the RAG loop. It gets the top-k chunks from Rust, puts them in the latest user turn as numbered `<source>` blocks, and streams Claude's answer with `@anthropic-ai/sdk`. Thinking is adaptive with summaries shown, effort is chosen in the UI, and `fallbacks: "default"` lets the API retry a refused request on another model.
- `server/prompt.ts`: system prompt and source formatting (answers cite `[n]`).
- `web/`: React UI with a model picker (Sonnet 5.5 / Opus 5.5), retrieval mode and effort. Clicking a citation highlights its source, and each source shows its BM25 and vector ranks, so you can see what fusion did.

The server runs on Node's built-in TypeScript support (Node ≥ 23.6), so there's no build step for it.

## Setup notes for this machine

- There's no MSVC linker here, so the folder is pinned to the GNU toolchain
  (`rustup override set stable-x86_64-pc-windows-gnu`), using WinLibs MinGW (installed with winget, on PATH).
  rustup stores the override by **folder path**, not inside the repo. If you rename or move
  the folder, or clone it somewhere else, run the command again in the new location.
  Otherwise cargo falls back to MSVC and fails with `linker 'link.exe' not found`.
  `rustup override list` shows which folders are pinned.
- `ort` has no prebuilt binaries for windows-gnu, so `vector-db` uses `load-dynamic`:
  `onnxruntime.dll` is loaded at runtime from `models/onnxruntime/` (or `$ORT_DYLIB_PATH`).
- HTTPS (crawler) uses `native-tls` (SChannel), so no C compiler or CMake is needed.
- Kaspersky's HTTPS scanning re-signs TLS traffic with its own root certificate, which
  Windows trusts but Node's built-in CA list doesn't (`SELF_SIGNED_CERT_IN_CHAIN`). The
  `rag` scripts run Node with `--use-system-ca` so it uses the Windows certificate store.
  Don't "fix" it with `NODE_TLS_REJECT_UNAUTHORIZED=0`, which turns off verification.
- Embedding is compute-bound: about 17 chunks/s (~7,500 tokens/s) in fp32 on this CPU.
  The int8 model (`Xenova/bge-small-en-v1.5`, `model_quantized.onnx`) would be about 2–3× faster.

## Ideas for next steps

- Query rewriting for follow-up questions (currently a heuristic in `retrieval.ts`)
- Re-ranking the fused top-20 with a cross-encoder before sending to Claude
- Incremental ingest (re-embed only changed files; HNSW supports inserts)
- Damerau-Levenshtein, phrase queries and an FST term dictionary in the search engine
- Prompt caching for long conversations

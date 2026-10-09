# mini-search

A small search engine built to learn how Google/Elasticsearch-style search works:
crawl a site (or index your own files), build an inverted index, rank with BM25,
and forgive typos with a Levenshtein automaton.

## Usage

```sh
# Crawl a site (stays under the start URL's directory; --whole-host to roam the site)
cargo run --release -p search-engine -- crawl https://doc.rust-lang.org/book/ --out index-book

# …or index your own projects and notes
cargo run --release -p search-engine -- index C:\Users\me\notes C:\Users\me\projetos --out index

# Search from the terminal (typo tolerance is on; --exact turns it off)
cargo run --release -p search-engine -- search -i index-book "borow chekcer lifetims"

# Web UI + JSON API at http://127.0.0.1:3000
cargo run --release -p search-engine -- serve -i index-book

# Compare the Levenshtein automaton against brute force for one word
cargo run --release -p search-engine -- fuzzy -i index-book refrence

cargo run --release -p search-engine -- stats -i index-book
```

API: `GET /api/search?q=…&limit=10&fuzzy=true`, `GET /api/fuzzy?term=…`, `GET /api/stats`.

## How it works

| Stage | File | Notes |
|---|---|---|
| Tokenizer | `src/tokenizer.rs` | split on non-alphanumerics, lowercase, fold accents (`Ação` → `acao`), drop stopwords |
| Stemmer | `src/stemmer.rs` | Porter (1980), English: `searching`/`searches` → `search` |
| Varint | `src/varint.rs` | LEB128: 7 bits per byte, high bit = "more follows" |
| Posting lists | `src/postings.rs` | doc-id gaps + term freqs, all varints; ~25% of plain `u32` pairs |
| Index | `src/index.rs` | sorted fixed-width term table → binary search directly over `mmap`ed files |
| BM25 | `src/bm25.rs` | k1 = 1.2, b = 0.75, Lucene's non-negative idf |
| Search | `src/search.rs` | per-query-word best expansion, blended idf for fuzzy terms, top-k selection, snippets |
| Typos | `src/levenshtein.rs` | DP distance → DFA built from DP rows → walked against the sorted terms, skipping dead prefixes |

Index files (`index/`): `terms.idx` (header + 24-byte entries), `terms.dat` (term strings),
`postings.dat`, `docs.dat` (stored text for snippets), `docs.json` (titles, URLs, lengths).

On a 329k-term dictionary (all of `~/.cargo/registry/src`), finding terms within 2 edits
took ~3.5 ms with the automaton vs ~235 ms brute force. It visited ~10% of the terms.

## Ideas for next steps

- Damerau-Levenshtein (transposition = 1 edit, as Lucene does): `distnace` → `distance`
- Positions in posting lists → phrase queries (`"borrow checker"`)
- Skip lists / block-max WAND to avoid scoring every matching doc
- A Portuguese stemmer (RSLP) alongside Porter
- An FST instead of a sorted array for the term dictionary (what Lucene/Tantivy use)

See the [workspace README](../../README.md) for how this crate fits with the vector DB and the RAG app.

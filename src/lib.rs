//! A small search engine, built to learn how Lucene/Elasticsearch-style search works.
//!
//! Hand-written (the interesting parts):
//! - [`tokenizer`] + [`stemmer`]: lowercase, accent folding, stopwords, Porter stemming
//! - [`varint`] + [`postings`]: delta + varint compressed posting lists
//! - [`index`]: the inverted index, written to flat files and memory-mapped
//! - [`bm25`] + [`search`]: ranking
//! - [`levenshtein`]: edit distance, then a Levenshtein automaton over the sorted terms
//!
//! Libraries do the plumbing: crawling ([`crawler`]), HTML parsing ([`html`]),
//! the HTTP API ([`server`]).

pub mod bm25;
pub mod crawler;
pub mod files;
pub mod html;
pub mod index;
pub mod levenshtein;
pub mod postings;
pub mod search;
pub mod server;
pub mod stemmer;
pub mod tokenizer;
pub mod varint;

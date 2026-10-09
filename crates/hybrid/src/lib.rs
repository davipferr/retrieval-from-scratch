//! Hybrid retrieval over your repos: the search engine's BM25 and the vector
//! DB's HNSW, run side by side and merged with Reciprocal Rank Fusion.
//!
//! Keyword search finds exact names (`FlowNodeView`, an error code) that
//! embeddings blur; embeddings find meaning ("where do we clean up
//! resources?" → `impl Drop`) that keywords miss. Fusing both is what
//! production RAG systems do.

pub mod chunk;
pub mod retriever;
pub mod rrf;
pub mod server;

//! A small vector database, built to learn how semantic search works.
//!
//! - [`vectors`] + [`flat`]: contiguous `f32` storage and the brute-force baseline
//! - [`distance`]: scalar vs `wide` SIMD vs AVX2 `std::arch` dot products
//! - [`hnsw`]: the graph index, with recall measured against brute force
//! - [`persist`] + [`collection`]: flat binary files, served through `mmap`
//! - [`embed`]: WordPiece tokenizer + ONNX Runtime for local embeddings

pub mod collection;
pub mod distance;
pub mod embed;
pub mod flat;
pub mod hnsw;
pub mod persist;
pub mod synthetic;
pub mod vectors;

use serde::Serialize;

pub use collection::{Collection, CollectionBuilder};

/// A search result: vector id and cosine similarity (1 = identical direction).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Neighbor {
    pub id: u32,
    pub score: f32,
}

/// Where the ONNX Runtime library lives: `$ORT_DYLIB_PATH`, or the copy that
/// `scripts/fetch-models.ps1` downloads.
pub fn default_ort_path() -> std::path::PathBuf {
    match std::env::var_os("ORT_DYLIB_PATH") {
        Some(p) if !p.is_empty() => p.into(),
        _ if cfg!(windows) => "models/onnxruntime/onnxruntime.dll".into(),
        _ if cfg!(target_os = "macos") => "models/onnxruntime/libonnxruntime.dylib".into(),
        _ => "models/onnxruntime/libonnxruntime.so".into(),
    }
}

//! Text → vector. An embedding model maps text to a point in a 384-dimensional
//! space where similar meanings land close together, so "how do I free memory"
//! ends up near a chunk about `Drop`, even though the two share no words.

pub mod onnx;
pub mod wordpiece;

use anyhow::Result;

pub trait Embedder: Send + Sync {
    /// Model name, recorded with the vectors it produced.
    fn name(&self) -> &str;
    fn dim(&self) -> usize;
    /// Embeds passages to be stored (chunks of documents).
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// Embeds a search query. Some models (like bge) want queries phrased
    /// differently from documents, so this is a separate method.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
}

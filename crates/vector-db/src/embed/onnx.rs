//! Local embeddings with ONNX Runtime (via `ort`): free, offline, in-process.
//!
//! A model directory holds `model.onnx` (the transformer) and `vocab.txt`
//! (for [`WordPiece`]). The model outputs one 384-d vector per *token*;
//! pooling turns those into one vector per *text*:
//!
//! - **CLS pooling** (bge): take the vector of the `[CLS]` token, which the
//!   model was trained to use as a summary of the whole sequence.
//! - **Mean pooling** (all-MiniLM): average all token vectors, ignoring padding.
//!
//! The result is L2-normalized so that dot product = cosine similarity.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, ensure};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use super::Embedder;
use super::wordpiece::WordPiece;
use crate::distance;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    Cls,
    Mean,
}

/// Texts per forward pass.
const BATCH: usize = 32;

pub struct OnnxEmbedder {
    // `Session::run` needs `&mut`, so concurrent callers take turns.
    session: Mutex<Session>,
    tokenizer: WordPiece,
    has_token_type_ids: bool,
    pooling: Pooling,
    /// bge recommends this instruction in front of short retrieval queries.
    query_prefix: &'static str,
    max_len: usize,
    dim: usize,
    pub model_name: String,
}

impl OnnxEmbedder {
    /// Loads `model_dir/model.onnx` + `model_dir/vocab.txt`.
    ///
    /// `ort_dylib` is the path to `onnxruntime.dll` / `libonnxruntime.so`
    /// (ONNX Runtime is loaded at runtime rather than linked; see the README).
    pub fn load(model_dir: &Path, ort_dylib: &Path) -> Result<Self> {
        ensure!(
            ort_dylib.exists(),
            "ONNX Runtime not found at {}",
            ort_dylib.display()
        );
        // Only the first call loads the library; later calls are no-ops.
        ort::init_from(ort_dylib)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit();

        let model_path = model_dir.join("model.onnx");
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort::Error::<()>::from)?
            .with_intra_threads(intra_threads())
            .map_err(ort::Error::<()>::from)?
            .commit_from_file(&model_path)
            .with_context(|| format!("loading {}", model_path.display()))?;
        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        let model_name = model_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let is_bge = model_name.to_lowercase().contains("bge");
        let mut embedder = Self {
            session: Mutex::new(session),
            tokenizer: WordPiece::from_file(&model_dir.join("vocab.txt"))?,
            has_token_type_ids,
            pooling: if is_bge { Pooling::Cls } else { Pooling::Mean },
            query_prefix: if is_bge {
                "Represent this sentence for searching relevant passages: "
            } else {
                ""
            },
            max_len: 512,
            dim: 0,
            model_name,
        };
        embedder.dim = embedder.run(&["probe"])?[0].len();
        Ok(embedder)
    }

    pub fn tokenizer(&self) -> &WordPiece {
        &self.tokenizer
    }

    /// One forward pass over a batch.
    fn run(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encoded: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| self.tokenizer.encode(t, self.max_len))
            .collect();
        let batch = encoded.len();
        let seq = encoded.iter().map(Vec::len).max().unwrap_or(0);

        // Pad every sequence to the longest; the attention mask tells the model
        // which positions are real (1) and which are padding (0).
        let mut ids = vec![0i64; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        for (b, tokens) in encoded.iter().enumerate() {
            for (s, &t) in tokens.iter().enumerate() {
                ids[b * seq + s] = i64::from(t);
                mask[b * seq + s] = 1;
            }
        }

        let shape = [batch as i64, seq as i64];
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array((shape, ids))?,
            "attention_mask" => Tensor::from_array((shape, mask.clone()))?,
        ];
        if self.has_token_type_ids {
            // single-segment input: all zeros
            inputs.push((
                "token_type_ids".into(),
                Tensor::from_array((shape, vec![0i64; batch * seq]))?.into(),
            ));
        }

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(inputs)?;
        // last_hidden_state: [batch, seq, hidden]
        let (out_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        ensure!(
            out_shape.len() == 3,
            "unexpected output shape {out_shape:?}"
        );
        let hidden = out_shape[2] as usize;

        let mut vectors = Vec::with_capacity(batch);
        for b in 0..batch {
            let token = |s: usize| &data[(b * seq + s) * hidden..(b * seq + s + 1) * hidden];
            let mut v = match self.pooling {
                Pooling::Cls => token(0).to_vec(),
                Pooling::Mean => {
                    let mut sum = vec![0f32; hidden];
                    let real = encoded[b].len();
                    for s in 0..real {
                        for (acc, x) in sum.iter_mut().zip(token(s)) {
                            *acc += x;
                        }
                    }
                    sum.iter_mut().for_each(|x| *x /= real as f32);
                    sum
                }
            };
            distance::normalize(&mut v);
            vectors.push(v);
        }
        Ok(vectors)
    }
}

/// Threads for one forward pass: `$EMBED_THREADS`, or ONNX Runtime's default.
fn intra_threads() -> usize {
    std::env::var("EMBED_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

impl Embedder for OnnxEmbedder {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Batch texts of similar length together: padding is wasted compute,
        // and attention cost grows with the square of the sequence length.
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| texts[i].len());
        let mut out = vec![Vec::new(); texts.len()];
        for chunk in order.chunks(BATCH) {
            let batch: Vec<&str> = chunk.iter().map(|&i| texts[i]).collect();
            for (&i, v) in chunk.iter().zip(self.run(&batch)?) {
                out[i] = v;
            }
        }
        Ok(out)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{}{text}", self.query_prefix);
        Ok(self.run(&[&prefixed])?.remove(0))
    }
}

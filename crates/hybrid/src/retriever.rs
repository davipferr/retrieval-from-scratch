//! Ingest: files → chunks → both indexes. Search: BM25, vectors, or both fused.
//!
//! ```text
//! data/
//!   bm25/         search-engine inverted index (one "document" per chunk)
//!   vectors/      vector-db collection (one vector per chunk)
//!   chunks.json   where each chunk came from
//! ```
//!
//! Chunk `i` is document `i` in the inverted index *and* vector `i` in the
//! collection, so results from the two engines can be joined by id.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use search_engine::files;
use search_engine::index::{Index, IndexBuilder};
use search_engine::search::{self as bm25_search, SearchOptions};
use vector_db::embed::Embedder;
use vector_db::hnsw::HnswParams;
use vector_db::{Collection, CollectionBuilder};

use crate::chunk;
use crate::rrf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMeta {
    /// Path shown to users: the ingested folder's name + the path inside it.
    pub path: String,
    pub abs_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub heading: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct IngestReport {
    pub files: usize,
    pub chunks: usize,
    pub embed_secs: f64,
    pub hnsw_secs: f64,
}

/// Chunks, embeds and indexes every text file under `roots`.
/// Paths containing any of `exclude` are skipped.
pub fn ingest(
    roots: &[PathBuf],
    data_dir: &Path,
    embedder: &dyn Embedder,
    exclude: &[String],
    mut progress: impl FnMut(&str),
) -> Result<IngestReport> {
    let mut metas: Vec<ChunkMeta> = Vec::new();
    let mut embed_texts: Vec<String> = Vec::new();
    let mut bm25 = IndexBuilder::new();
    let mut n_files = 0;

    for root in roots {
        let root = fs::canonicalize(root).with_context(|| format!("reading {}", root.display()))?;
        let label = root
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        for file in files::collect(&root)? {
            let rel = file.path.strip_prefix(&root).unwrap_or(&file.path);
            let display = format!("{label}/{}", rel.display()).replace('\\', "/");
            if exclude.iter().any(|x| display.contains(x.as_str())) {
                continue;
            }
            n_files += 1;
            let markdown = matches!(
                file.path.extension().and_then(|e| e.to_str()),
                Some("md" | "markdown")
            );
            for c in chunk::chunk_text(&file.text, markdown) {
                let title = match &c.heading {
                    Some(h) => format!("{display} › {h}"),
                    None => display.clone(),
                };
                // The path is indexed as the title (boosted in BM25) and
                // prepended for embedding: `auth/session.ts` says a lot about
                // what the code inside is for.
                bm25.add_document(
                    &format!("{display}:{}-{}", c.start_line, c.end_line),
                    &title,
                    &c.text,
                );
                embed_texts.push(format!("{title}\n{}", c.text));
                metas.push(ChunkMeta {
                    path: display.clone(),
                    abs_path: file.path.display().to_string(),
                    start_line: c.start_line,
                    end_line: c.end_line,
                    heading: c.heading,
                });
            }
        }
    }
    progress(&format!("{n_files} files -> {} chunks", metas.len()));

    fs::create_dir_all(data_dir)?;
    bm25.write(&data_dir.join("bm25"))?;

    let started = Instant::now();
    let mut collection = CollectionBuilder::new(embedder.dim(), Some(embedder.name().to_string()));
    for (i, batch) in embed_texts.chunks(256).enumerate() {
        let refs: Vec<&str> = batch.iter().map(String::as_str).collect();
        for v in embedder.embed_documents(&refs)? {
            collection.push(&v);
        }
        let done = (i * 256 + batch.len()).min(embed_texts.len());
        let rate = done as f64 / started.elapsed().as_secs_f64();
        progress(&format!(
            "embedded {done}/{} ({rate:.0} chunks/s)",
            embed_texts.len()
        ));
    }
    let embed_secs = started.elapsed().as_secs_f64();

    let report = collection.write(&data_dir.join("vectors"), HnswParams::default(), |_| {})?;
    fs::write(data_dir.join("chunks.json"), serde_json::to_vec(&metas)?)?;

    Ok(IngestReport {
        files: n_files,
        chunks: metas.len(),
        embed_secs,
        hnsw_secs: report.build_secs,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Bm25,
    Vector,
    #[default]
    Hybrid,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub id: u32,
    #[serde(flatten)]
    pub chunk: ChunkMeta,
    pub text: String,
    /// The score the final ranking used (RRF in hybrid mode).
    pub score: f32,
    /// 1-based rank in each engine's own list, if the chunk appeared there.
    pub bm25_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub bm25_score: Option<f32>,
    pub similarity: Option<f32>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct Timings {
    pub bm25_ms: f64,
    pub embed_ms: f64,
    pub vector_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub mode: Mode,
    pub hits: Vec<Hit>,
    pub timings: Timings,
}

pub struct Retriever {
    bm25: Index,
    vectors: Collection,
    embedder: Box<dyn Embedder>,
    chunks: Vec<ChunkMeta>,
}

/// How many results each engine contributes before fusion.
const CANDIDATES: usize = 50;

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

impl Retriever {
    pub fn open(data_dir: &Path, embedder: Box<dyn Embedder>) -> Result<Self> {
        let bm25 = Index::open(&data_dir.join("bm25"))?;
        let vectors = Collection::open(&data_dir.join("vectors"))?;
        let chunks: Vec<ChunkMeta> = serde_json::from_slice(
            &fs::read(data_dir.join("chunks.json")).context("reading chunks.json")?,
        )?;
        ensure!(
            bm25.num_docs() == chunks.len() && vectors.len() == chunks.len(),
            "data dir is inconsistent: re-run ingest"
        );
        ensure!(
            vectors.dim() == embedder.dim()
                && vectors
                    .meta
                    .model
                    .as_deref()
                    .is_none_or(|m| m == embedder.name()),
            "collection was built with a different embedding model ({:?})",
            vectors.meta.model
        );
        Ok(Self {
            bm25,
            vectors,
            embedder,
            chunks,
        })
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn files(&self) -> usize {
        let mut paths: Vec<&str> = self.chunks.iter().map(|c| c.path.as_str()).collect();
        paths.sort_unstable();
        paths.dedup();
        paths.len()
    }

    pub fn dim(&self) -> usize {
        self.vectors.dim()
    }

    pub fn search(&self, query: &str, k: usize, mode: Mode) -> Result<SearchResponse> {
        let started = Instant::now();
        let mut timings = Timings::default();

        let bm25_list: Vec<(u32, f32)> = if mode != Mode::Vector {
            let t = Instant::now();
            let opts = SearchOptions {
                limit: CANDIDATES,
                fuzzy: true,
                ..Default::default()
            };
            let r = bm25_search::search(&self.bm25, query, &opts);
            timings.bm25_ms = ms(t);
            r.hits.iter().map(|h| (h.doc, h.score)).collect()
        } else {
            Vec::new()
        };

        let vector_list: Vec<(u32, f32)> = if mode != Mode::Bm25 {
            let t = Instant::now();
            let q = self.embedder.embed_query(query)?;
            timings.embed_ms = ms(t);
            let t = Instant::now();
            let (hits, _) = self.vectors.search(&q, CANDIDATES, CANDIDATES.max(64));
            timings.vector_ms = ms(t);
            hits.iter().map(|n| (n.id, n.score)).collect()
        } else {
            Vec::new()
        };

        let ranked: Vec<(u32, f32)> = match mode {
            Mode::Bm25 => bm25_list.clone(),
            Mode::Vector => vector_list.clone(),
            Mode::Hybrid => {
                let a: Vec<u32> = bm25_list.iter().map(|x| x.0).collect();
                let b: Vec<u32> = vector_list.iter().map(|x| x.0).collect();
                rrf::fuse(&[&a, &b], rrf::DEFAULT_K)
            }
        };

        let find = |list: &[(u32, f32)], id: u32| {
            list.iter()
                .position(|x| x.0 == id)
                .map(|p| (p + 1, list[p].1))
        };
        let hits = ranked
            .into_iter()
            .take(k)
            .map(|(id, score)| {
                let bm25 = find(&bm25_list, id);
                let vector = find(&vector_list, id);
                Hit {
                    id,
                    chunk: self.chunks[id as usize].clone(),
                    text: self.bm25.doc_text(id).to_string(),
                    score,
                    bm25_rank: bm25.map(|x| x.0),
                    vector_rank: vector.map(|x| x.0),
                    bm25_score: bm25.map(|x| x.1),
                    similarity: vector.map(|x| x.1),
                }
            })
            .collect();

        timings.total_ms = ms(started);
        Ok(SearchResponse {
            query: query.to_string(),
            mode,
            hits,
            timings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in embedder: hashes words into buckets ("bag of words"), so
    /// tests don't need the ONNX model. Texts sharing words end up similar.
    struct HashEmbedder;

    impl HashEmbedder {
        fn vec(text: &str) -> Vec<f32> {
            let mut v = vec![0.0; 64];
            for w in search_engine::tokenizer::analyze(text) {
                let h = w
                    .bytes()
                    .fold(5381u32, |h, b| h.wrapping_mul(33) ^ u32::from(b));
                v[h as usize % 64] += 1.0;
            }
            v[63] += 0.01; // never all-zero
            v
        }
    }

    impl Embedder for HashEmbedder {
        fn name(&self) -> &str {
            "hash"
        }
        fn dim(&self) -> usize {
            64
        }
        fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| Self::vec(t)).collect())
        }
        fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            Ok(Self::vec(text))
        }
    }

    #[test]
    fn ingest_and_search_all_modes() {
        let base = std::env::temp_dir().join(format!("hybrid_test_{}", std::process::id()));
        let repo = base.join("repo");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("src/FlowNodeView.tsx"),
            "export function FlowNodeView() {\n  return <div/>;\n}\n",
        )
        .unwrap();
        fs::write(
            repo.join("README.md"),
            "# Repo\nA graph editor.\n\n## Testing\nRun the unit tests with npm.\n",
        )
        .unwrap();
        fs::write(repo.join("skip.txt"), "excluded file about FlowNodeView\n").unwrap();

        let data = base.join("data");
        let report = ingest(
            std::slice::from_ref(&repo),
            &data,
            &HashEmbedder,
            &["skip.txt".into()],
            |_| {},
        )
        .unwrap();
        assert_eq!(report.files, 2);

        let r = Retriever::open(&data, Box::new(HashEmbedder)).unwrap();
        for mode in [Mode::Bm25, Mode::Vector, Mode::Hybrid] {
            let res = r.search("FlowNodeView", 3, mode).unwrap();
            assert_eq!(
                res.hits[0].chunk.path, "repo/src/FlowNodeView.tsx",
                "{mode:?}"
            );
            assert_eq!(
                (res.hits[0].chunk.start_line, res.hits[0].chunk.end_line),
                (1, 3)
            );
        }
        let res = r.search("unit tests", 1, Mode::Hybrid).unwrap();
        assert_eq!(res.hits[0].chunk.heading.as_deref(), Some("Testing"));
        assert!(res.hits[0].bm25_rank.is_some() && res.hits[0].vector_rank.is_some());
        drop(r);
        fs::remove_dir_all(base).ok();
    }
}

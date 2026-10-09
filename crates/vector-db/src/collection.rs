//! A collection: vectors + HNSW graph + metadata in one directory.
//!
//! ```text
//! collection/
//!   vectors.bin   flat f32 vectors (see `persist`)
//!   hnsw.bin      the graph, CSR layout
//!   meta.json     dimension, count, model name, HNSW parameters
//! ```
//!
//! Built once in memory, written, then served read-only through mmap.
//! Vector ids are their insertion order, 0..n.

use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::Neighbor;
use crate::distance;
use crate::flat;
use crate::hnsw::{self, HnswParams, SearchStats};
use crate::persist::{self, MmapGraph, MmapVectors};
use crate::vectors::{VecStore, VectorSource};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMeta {
    pub dim: usize,
    pub count: usize,
    pub model: Option<String>,
    pub hnsw: HnswParams,
}

pub struct CollectionBuilder {
    store: VecStore,
    model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BuildReport {
    pub count: usize,
    pub build_secs: f64,
    pub vectors_bytes: u64,
    pub graph_bytes: u64,
}

impl CollectionBuilder {
    pub fn new(dim: usize, model: Option<String>) -> Self {
        Self {
            store: VecStore::new(dim),
            model,
        }
    }

    pub fn push(&mut self, v: &[f32]) -> u32 {
        self.store.push(v)
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Builds the HNSW graph and writes everything to `dir`.
    pub fn write(
        self,
        dir: &Path,
        params: HnswParams,
        progress: impl FnMut(usize),
    ) -> Result<BuildReport> {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let started = Instant::now();
        let graph = hnsw::build(&self.store, params, progress);
        let build_secs = started.elapsed().as_secs_f64();

        let (vectors_path, graph_path) = (dir.join("vectors.bin"), dir.join("hnsw.bin"));
        persist::write_vectors(&self.store, &vectors_path)?;
        persist::write_graph(&graph, &graph_path)?;
        let meta = CollectionMeta {
            dim: self.store.dim(),
            count: self.store.len(),
            model: self.model,
            hnsw: params,
        };
        fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;

        Ok(BuildReport {
            count: meta.count,
            build_secs,
            vectors_bytes: persist::file_size(&vectors_path),
            graph_bytes: persist::file_size(&graph_path),
        })
    }
}

pub struct Collection {
    vectors: MmapVectors,
    graph: MmapGraph,
    pub meta: CollectionMeta,
}

impl Collection {
    pub fn open(dir: &Path) -> Result<Self> {
        let meta: CollectionMeta = serde_json::from_slice(
            &fs::read(dir.join("meta.json"))
                .with_context(|| format!("no collection in {}", dir.display()))?,
        )?;
        let vectors = MmapVectors::open(&dir.join("vectors.bin"))?;
        let graph = MmapGraph::open(&dir.join("hnsw.bin"))?;
        ensure!(
            vectors.len() == graph.len() && vectors.dim() == meta.dim,
            "collection files disagree"
        );
        Ok(Self {
            vectors,
            graph,
            meta,
        })
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.vectors.dim()
    }

    pub fn vector(&self, id: u32) -> &[f32] {
        self.vectors.get(id)
    }

    fn unit(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim(), "query has the wrong dimension");
        let mut q = query.to_vec();
        distance::normalize(&mut q);
        q
    }

    /// Approximate top-k with HNSW.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> (Vec<Neighbor>, SearchStats) {
        hnsw::search(&self.graph, &self.vectors, &self.unit(query), k, ef)
    }

    /// Exact top-k by brute force.
    pub fn search_exact(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        flat::search(&self.vectors, &self.unit(query), k)
    }

    pub fn recall(&self, queries: &[Vec<f32>], k: usize, ef: usize) -> f64 {
        hnsw::recall(&self.graph, &self.vectors, queries, k, ef)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::clustered;

    #[test]
    fn write_open_search() {
        let data = clustered(800, 16, 4);
        let mut builder = CollectionBuilder::new(16, Some("test".into()));
        for id in 0..data.len() as u32 {
            builder.push(data.get(id));
        }
        let dir = std::env::temp_dir().join(format!("vecdb_collection_{}", std::process::id()));
        let report = builder.write(&dir, HnswParams::default(), |_| {}).unwrap();
        assert_eq!(report.count, 800);

        let c = Collection::open(&dir).unwrap();
        assert_eq!(c.len(), 800);
        // a scaled copy of a stored vector must find that vector
        let q: Vec<f32> = c.vector(123).iter().map(|x| x * 3.0).collect();
        assert_eq!(c.search(&q, 1, 32).0[0].id, 123);
        assert_eq!(c.search_exact(&q, 1)[0].id, 123);
        drop(c);
        fs::remove_dir_all(dir).ok();
    }
}

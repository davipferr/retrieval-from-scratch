//! On-disk formats, read back through `mmap` without deserializing anything.
//!
//! The trick (same as the search engine's term table): lay data out on disk
//! exactly as it should look in memory, with correct alignment, so that after
//! mapping a file a `&[u8]` can be reinterpreted as `&[f32]` / `&[u32]` in
//! place. Opening a 1 GB index is then instant, and the OS loads only the pages
//! a search touches.
//!
//! `vectors.bin`
//! ```text
//! magic "MVDBVEC1" | dim u32 | reserved u32 | f32 × (n · dim)
//! ```
//!
//! `hnsw.bin` — the graph in CSR ("compressed sparse row") form: instead of a
//! `Vec<Vec<Vec<u32>>>` with a heap allocation per list, one flat array of
//! all neighbor ids plus an offsets array saying where each list starts.
//! ```text
//! header (32 bytes): magic "MVDBHNS1" | n u32 | entry u32 | max_level u32 | m u32 | m0 u32 | reserved u32
//! node_slot:   u32 × (n + 1)       node i's lists are slots node_slot[i] .. node_slot[i+1]
//!                                  (one slot per level, so the count is level + 1)
//! slot_offset: u32 × (slots + 1)   slot s's neighbors are neighbors[slot_offset[s] .. slot_offset[s+1]]
//! neighbors:   u32 × total
//! ```
//! Everything is little-endian, which is what x86 and ARM use natively.

#[cfg(target_endian = "big")]
compile_error!("the on-disk formats assume a little-endian CPU");

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;

use crate::hnsw::{Graph, HnswGraph};
use crate::vectors::{VecStore, VectorSource};

const VEC_MAGIC: &[u8; 8] = b"MVDBVEC1";
const VEC_HEADER: usize = 16;
const GRAPH_MAGIC: &[u8; 8] = b"MVDBHNS1";
const GRAPH_HEADER: usize = 32;
const NO_ENTRY: u32 = u32::MAX;

fn map(path: &Path) -> Result<Mmap> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // SAFETY: index files are written once and treated as immutable afterwards.
    // (If something truncated the file while mapped, reads could fault.)
    unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))
}

/// Reinterprets bytes as a slice of 4-byte plain-old-data values.
fn cast<T: Pod4>(bytes: &[u8]) -> &[T] {
    assert!(bytes.len().is_multiple_of(4) && (bytes.as_ptr() as usize).is_multiple_of(4));
    // SAFETY: length and alignment were just checked, and every bit pattern is
    // a valid u32/f32 (that's what the `Pod4` marker promises).
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), bytes.len() / 4) }
}

/// Types for which any 4 bytes are a valid value.
trait Pod4: Copy {}
impl Pod4 for u32 {}
impl Pod4 for f32 {}

fn u32_at(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(buf[at..at + 4].try_into().unwrap())
}

fn write_u32s(w: &mut impl Write, values: impl IntoIterator<Item = u32>) -> std::io::Result<()> {
    for v in values {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------

pub fn write_vectors(store: &VecStore, path: &Path) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(VEC_MAGIC)?;
    w.write_all(&(store.dim() as u32).to_le_bytes())?;
    w.write_all(&0u32.to_le_bytes())?;
    for x in store.as_slice() {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

pub struct MmapVectors {
    map: Mmap,
    dim: usize,
    len: usize,
}

impl MmapVectors {
    pub fn open(path: &Path) -> Result<Self> {
        let map = map(path)?;
        ensure!(
            map.len() >= VEC_HEADER && &map[..8] == VEC_MAGIC,
            "{}: not a vector file",
            path.display()
        );
        let dim = u32_at(&map, 8) as usize;
        let body = map.len() - VEC_HEADER;
        ensure!(
            dim > 0 && body.is_multiple_of(dim * 4),
            "{}: truncated",
            path.display()
        );
        Ok(Self {
            len: body / (dim * 4),
            dim,
            map,
        })
    }

    fn data(&self) -> &[f32] {
        // mmap returns page-aligned memory and the header is 16 bytes, so this is aligned.
        cast(&self.map[VEC_HEADER..])
    }
}

impl VectorSource for MmapVectors {
    fn dim(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        &self.data()[start..start + self.dim]
    }
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

pub fn write_graph(graph: &HnswGraph, path: &Path) -> Result<()> {
    let n = graph.len();
    let mut node_slot = Vec::with_capacity(n + 1);
    let mut slot_offset = vec![0u32];
    let mut neighbors: Vec<u32> = Vec::new();
    for node in 0..n as u32 {
        node_slot.push(slot_offset.len() as u32 - 1);
        for level in 0..=graph.node_level(node) {
            neighbors.extend_from_slice(graph.neighbors(node, level));
            slot_offset.push(neighbors.len() as u32);
        }
    }
    node_slot.push(slot_offset.len() as u32 - 1);

    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(GRAPH_MAGIC)?;
    let p = graph.params;
    let header = [
        n as u32,
        graph.entry_point().unwrap_or(NO_ENTRY),
        graph.max_level() as u32,
        p.m as u32,
        p.m0 as u32,
        0,
    ];
    write_u32s(&mut w, header)?;
    write_u32s(&mut w, node_slot)?;
    write_u32s(&mut w, slot_offset)?;
    write_u32s(&mut w, neighbors)?;
    w.flush()?;
    Ok(())
}

/// An HNSW graph read straight out of a memory map.
pub struct MmapGraph {
    map: Mmap,
    entry: Option<u32>,
    max_level: usize,
    node_slot: Range<usize>,
    slot_offset: Range<usize>,
    neighbors: Range<usize>,
}

impl MmapGraph {
    pub fn open(path: &Path) -> Result<Self> {
        let map = map(path)?;
        ensure!(
            map.len() >= GRAPH_HEADER && &map[..8] == GRAPH_MAGIC,
            "{}: not a graph file",
            path.display()
        );
        let n = u32_at(&map, 8) as usize;
        let entry = u32_at(&map, 12);
        let max_level = u32_at(&map, 16) as usize;

        let node_slot = GRAPH_HEADER..GRAPH_HEADER + 4 * (n + 1);
        ensure!(map.len() >= node_slot.end, "graph truncated");
        let slots = *cast::<u32>(&map[node_slot.clone()]).last().unwrap() as usize;
        let slot_offset = node_slot.end..node_slot.end + 4 * (slots + 1);
        ensure!(map.len() >= slot_offset.end, "graph truncated");
        let total = *cast::<u32>(&map[slot_offset.clone()]).last().unwrap() as usize;
        let neighbors = slot_offset.end..slot_offset.end + 4 * total;
        ensure!(map.len() == neighbors.end, "graph has the wrong length");

        let graph = Self {
            entry: (entry != NO_ENTRY).then_some(entry),
            max_level,
            node_slot,
            slot_offset,
            neighbors,
            map,
        };
        graph.validate(n)?;
        Ok(graph)
    }

    /// Checks every offset once, so `neighbors()` can't index out of bounds later.
    fn validate(&self, n: usize) -> Result<()> {
        let node_slot = cast::<u32>(&self.map[self.node_slot.clone()]);
        let slot_offset = cast::<u32>(&self.map[self.slot_offset.clone()]);
        let neighbors = cast::<u32>(&self.map[self.neighbors.clone()]);
        if node_slot.windows(2).any(|w| w[0] >= w[1]) || slot_offset.windows(2).any(|w| w[0] > w[1])
        {
            bail!("graph offsets are not increasing");
        }
        ensure!(
            neighbors.iter().all(|&nb| (nb as usize) < n),
            "neighbor id out of range"
        );
        if let Some(e) = self.entry {
            ensure!((e as usize) < n, "entry point out of range");
            let entry_levels = node_slot[e as usize + 1] - node_slot[e as usize];
            ensure!(
                entry_levels as usize == self.max_level + 1,
                "entry point is not on the top level"
            );
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.node_slot.len() / 4 - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Graph for MmapGraph {
    fn entry_point(&self) -> Option<u32> {
        self.entry
    }
    fn max_level(&self) -> usize {
        self.max_level
    }
    fn neighbors(&self, node: u32, level: usize) -> &[u32] {
        let node_slot = cast::<u32>(&self.map[self.node_slot.clone()]);
        let slot_offset = cast::<u32>(&self.map[self.slot_offset.clone()]);
        let slot = node_slot[node as usize] as usize + level;
        debug_assert!(
            slot < node_slot[node as usize + 1] as usize,
            "node not on this level"
        );
        let (start, end) = (slot_offset[slot] as usize, slot_offset[slot + 1] as usize);
        &cast::<u32>(&self.map[self.neighbors.clone()])[start..end]
    }
}

pub fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::{self, HnswParams};

    #[test]
    fn roundtrip_through_mmap() {
        let store = crate::synthetic::clustered(1500, 24, 9);
        let graph = hnsw::build(&store, HnswParams::default(), |_| {});
        let dir = std::env::temp_dir().join(format!("vecdb_persist_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        write_vectors(&store, &dir.join("vectors.bin")).unwrap();
        write_graph(&graph, &dir.join("hnsw.bin")).unwrap();

        let vectors = MmapVectors::open(&dir.join("vectors.bin")).unwrap();
        let mapped = MmapGraph::open(&dir.join("hnsw.bin")).unwrap();
        assert_eq!(vectors.len(), store.len());
        assert_eq!(vectors.get(77), store.get(77));
        assert_eq!(mapped.len(), graph.len());
        assert_eq!(mapped.entry_point(), graph.entry_point());
        for node in 0..graph.len() as u32 {
            for level in 0..=graph.node_level(node) {
                assert_eq!(mapped.neighbors(node, level), graph.neighbors(node, level));
            }
        }
        // same graph, same vectors => identical search results
        let q = store.get(5);
        assert_eq!(
            hnsw::search(&mapped, &vectors, q, 10, 50).0,
            hnsw::search(&graph, &store, q, 10, 50).0
        );
        drop((vectors, mapped));
        fs::remove_dir_all(dir).ok();
    }
}

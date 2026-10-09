//! HNSW: Hierarchical Navigable Small World graphs
//! (Malkov & Yashunin, 2016 — <https://arxiv.org/abs/1603.09320>).
//!
//! Every vector is a node in a graph, linked to a handful of its nearest
//! neighbors. To search, start somewhere and greedily walk to whichever
//! neighbor is closer to the query, until nothing closer is left.
//!
//! On one flat graph, that walk takes many short steps. HNSW adds *layers*,
//! like a skip list:
//!
//! ```text
//! layer 2:  A ─────────────────────── F            few nodes, long links
//! layer 1:  A ──────── C ──────────── F ──── H
//! layer 0:  A ── B ── C ── D ── E ── F ── G ── H   every node, short links
//! ```
//!
//! Each node gets a random top layer (geometric distribution: ~1/M of the
//! nodes reach layer 1, 1/M² layer 2…). Search enters at the top, takes a few
//! long hops to get near the query, then descends and refines. That makes
//! search roughly O(log n) distance computations instead of O(n).
//!
//! The price: results are *approximate*. A greedy walk can miss the true
//! nearest neighbor, so we measure **recall** — the fraction of the exact
//! top-k (from brute force) that HNSW returns — and trade it against speed
//! with `ef`, the size of the candidate list kept during search.

use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use crate::Neighbor;
use crate::distance::cosine_distance;
use crate::flat;
use crate::vectors::VectorSource;

/// Read access to an HNSW graph, in memory or memory-mapped.
pub trait Graph: Sync {
    fn entry_point(&self) -> Option<u32>;
    fn max_level(&self) -> usize;
    /// Neighbors of `node` on `level` (the node must exist on that level).
    fn neighbors(&self, node: u32, level: usize) -> &[u32];
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HnswParams {
    /// Links per node on layers ≥ 1 (and how many a new node picks on each layer).
    pub m: usize,
    /// Max links per node on layer 0, which holds every node; usually 2·M.
    pub m0: usize,
    /// Candidate list size while inserting. Higher = better graph, slower build.
    pub ef_construction: usize,
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            seed: 0x5eed,
        }
    }
}

/// A node id with its distance to the current query, ordered by distance.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cand {
    dist: f32,
    id: u32,
}

impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then(self.id.cmp(&other.id))
    }
}

/// Counters for one search, to see how much work HNSW avoided.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct SearchStats {
    pub distance_evals: usize,
    /// Nodes whose neighbor lists were expanded.
    pub hops: usize,
}

/// "Have I seen this node during the current search?" A plain bitmap would
/// need clearing (O(n)) before every search; instead each slot stores the
/// number of the search that last saw it, and starting a new search just
/// increments the counter.
struct Visited {
    marks: Vec<u32>,
    epoch: u32,
}

impl Visited {
    fn reset(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    /// Marks `id`; returns `false` if it was already marked.
    fn insert(&mut self, id: u32) -> bool {
        let slot = &mut self.marks[id as usize];
        if *slot == self.epoch {
            return false;
        }
        *slot = self.epoch;
        true
    }
}

thread_local! {
    // One per thread, reused across searches so the server doesn't allocate per query.
    static VISITED: RefCell<Visited> = const { RefCell::new(Visited { marks: Vec::new(), epoch: 0 }) };
}

/// The core routine (Algorithm 2 in the paper): best-first search on one layer.
///
/// Keeps two heaps: `candidates` (nodes still to expand, closest first) and
/// `results` (the best `ef` found so far, farthest on top). It stops when the
/// closest unexpanded candidate is farther than the worst result: expanding it
/// could no longer improve anything. Returns results sorted closest first.
fn search_layer(
    graph: &impl Graph,
    vectors: &impl VectorSource,
    query: &[f32],
    entry: &[Cand],
    ef: usize,
    level: usize,
    stats: &mut SearchStats,
) -> Vec<Cand> {
    VISITED.with_borrow_mut(|visited| {
        visited.reset(vectors.len());
        let mut candidates: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        let mut results: BinaryHeap<Cand> = BinaryHeap::new();
        for &e in entry {
            if visited.insert(e.id) {
                candidates.push(Reverse(e));
                results.push(e);
            }
        }
        while results.len() > ef {
            results.pop();
        }

        while let Some(Reverse(current)) = candidates.pop() {
            let worst = results.peek().map_or(f32::INFINITY, |c| c.dist);
            if current.dist > worst && results.len() >= ef {
                break;
            }
            stats.hops += 1;
            for &nb in graph.neighbors(current.id, level) {
                if !visited.insert(nb) {
                    continue;
                }
                let dist = cosine_distance(query, vectors.get(nb));
                stats.distance_evals += 1;
                if results.len() < ef || dist < results.peek().unwrap().dist {
                    let c = Cand { dist, id: nb };
                    candidates.push(Reverse(c));
                    results.push(c);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        results.into_sorted_vec()
    })
}

/// Neighbor selection heuristic (Algorithm 4).
///
/// Taking simply the M closest candidates tends to pick a tight clump all on
/// one side of the node, which makes the graph poorly connected between
/// clusters. Instead, walk the candidates closest first and keep one only if
/// it is closer to the base node than to every neighbor kept so far — i.e. it
/// is not "behind" an already chosen neighbor. That spreads links out in
/// different directions.
fn select_neighbors(vectors: &impl VectorSource, sorted: &[Cand], m: usize) -> Vec<u32> {
    let mut kept: Vec<u32> = Vec::with_capacity(m);
    for c in sorted {
        if kept.len() >= m {
            break;
        }
        let v = vectors.get(c.id);
        if kept
            .iter()
            .all(|&k| cosine_distance(v, vectors.get(k)) > c.dist)
        {
            kept.push(c.id);
        }
    }
    kept
}

/// Tiny, fast, seedable PRNG (SplitMix64) for picking levels.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    pub(crate) fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }
}

/// An HNSW graph in memory, as produced by [`build`].
pub struct HnswGraph {
    pub params: HnswParams,
    /// `links[node][level]` = neighbor ids.
    links: Vec<Vec<Vec<u32>>>,
    entry: Option<u32>,
    max_level: usize,
}

impl Graph for HnswGraph {
    fn entry_point(&self) -> Option<u32> {
        self.entry
    }
    fn max_level(&self) -> usize {
        self.max_level
    }
    fn neighbors(&self, node: u32, level: usize) -> &[u32] {
        &self.links[node as usize][level]
    }
}

impl HnswGraph {
    pub fn len(&self) -> usize {
        self.links.len()
    }

    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Top layer of each node.
    pub fn node_level(&self, node: u32) -> usize {
        self.links[node as usize].len() - 1
    }

    /// Average number of links per node on `level`.
    pub fn avg_degree(&self, level: usize) -> f64 {
        let lists: Vec<usize> = self
            .links
            .iter()
            .filter(|l| l.len() > level)
            .map(|l| l[level].len())
            .collect();
        lists.iter().sum::<usize>() as f64 / lists.len().max(1) as f64
    }

    /// Inserts vector `id` (Algorithm 1). Ids must be inserted in order 0, 1, 2…
    fn insert(&mut self, vectors: &impl VectorSource, id: u32, level: usize) {
        debug_assert_eq!(id as usize, self.links.len());
        let p = self.params;
        let q = vectors.get(id);
        self.links.push(vec![Vec::new(); level + 1]);

        let Some(entry) = self.entry else {
            self.entry = Some(id);
            self.max_level = level;
            return;
        };

        let mut stats = SearchStats::default();
        let mut eps = vec![Cand {
            dist: cosine_distance(q, vectors.get(entry)),
            id: entry,
        }];

        // 1. Above the new node's level: just greedily find the closest node.
        for lc in (level + 1..=self.max_level).rev() {
            eps = search_layer(self, vectors, q, &eps, 1, lc, &mut stats);
        }

        // 2. On each of its levels: find ef_construction candidates, link to the best M.
        for lc in (0..=level.min(self.max_level)).rev() {
            let found = search_layer(self, vectors, q, &eps, p.ef_construction, lc, &mut stats);
            let selected = select_neighbors(vectors, &found, p.m);
            let cap = if lc == 0 { p.m0 } else { p.m };

            for &nb in &selected {
                let list = &mut self.links[nb as usize][lc];
                list.push(id);
                if list.len() > cap {
                    // Too many links now: re-run the heuristic from the neighbor's side.
                    let base = vectors.get(nb);
                    let mut cands: Vec<Cand> = list
                        .iter()
                        .map(|&x| Cand {
                            dist: cosine_distance(base, vectors.get(x)),
                            id: x,
                        })
                        .collect();
                    cands.sort_unstable();
                    self.links[nb as usize][lc] = select_neighbors(vectors, &cands, cap);
                }
            }
            self.links[id as usize][lc] = selected;
            eps = found;
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(id);
        }
    }
}

/// Builds a graph over every vector in `vectors`. `progress` is called every 1000 inserts.
pub fn build(
    vectors: &impl VectorSource,
    params: HnswParams,
    mut progress: impl FnMut(usize),
) -> HnswGraph {
    let mut graph = HnswGraph {
        params,
        links: Vec::with_capacity(vectors.len()),
        entry: None,
        max_level: 0,
    };
    let mut rng = Rng(params.seed);
    // P(level ≥ l) = M^-l, so level = floor(-ln(U) / ln(M)).
    let ml = 1.0 / (params.m as f64).ln();
    for id in 0..vectors.len() as u32 {
        let level = ((-rng.next_f64().ln() * ml).floor() as usize).min(16);
        graph.insert(vectors, id, level);
        if (id + 1) % 1000 == 0 {
            progress(id as usize + 1);
        }
    }
    graph
}

/// k-nearest-neighbor search (Algorithm 5).
pub fn search(
    graph: &impl Graph,
    vectors: &impl VectorSource,
    query: &[f32],
    k: usize,
    ef: usize,
) -> (Vec<Neighbor>, SearchStats) {
    let mut stats = SearchStats::default();
    let Some(entry) = graph.entry_point() else {
        return (Vec::new(), stats);
    };
    let mut eps = vec![Cand {
        dist: cosine_distance(query, vectors.get(entry)),
        id: entry,
    }];
    stats.distance_evals += 1;

    for lc in (1..=graph.max_level()).rev() {
        eps = search_layer(graph, vectors, query, &eps, 1, lc, &mut stats);
    }
    let found = search_layer(graph, vectors, query, &eps, ef.max(k), 0, &mut stats);
    let hits = found
        .into_iter()
        .take(k)
        .map(|c| Neighbor {
            id: c.id,
            score: 1.0 - c.dist,
        })
        .collect();
    (hits, stats)
}

/// Mean recall@k of HNSW against brute force over `queries`.
pub fn recall(
    graph: &impl Graph,
    vectors: &impl VectorSource,
    queries: &[Vec<f32>],
    k: usize,
    ef: usize,
) -> f64 {
    let mut total = 0.0;
    for q in queries {
        let exact: Vec<u32> = flat::search(vectors, q, k).iter().map(|n| n.id).collect();
        let (approx, _) = search(graph, vectors, q, k, ef);
        let found = approx.iter().filter(|n| exact.contains(&n.id)).count();
        total += found as f64 / exact.len().max(1) as f64;
    }
    total / queries.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::clustered;
    use crate::vectors::VecStore;

    #[test]
    fn recall_is_high() {
        let store = clustered(3000, 32, 1);
        let graph = build(&store, HnswParams::default(), |_| {});
        let queries: Vec<Vec<f32>> = (0..100).map(|i| store.get(i * 29).to_vec()).collect();
        let r = recall(&graph, &store, &queries, 10, 64);
        assert!(r > 0.95, "recall@10 = {r}");
    }

    #[test]
    fn degree_is_bounded() {
        let store = clustered(2000, 16, 2);
        let p = HnswParams {
            m: 8,
            m0: 16,
            ..Default::default()
        };
        let graph = build(&store, p, |_| {});
        for node in 0..graph.len() as u32 {
            assert!(graph.neighbors(node, 0).len() <= p.m0);
            for l in 1..=graph.node_level(node) {
                assert!(graph.neighbors(node, l).len() <= p.m);
            }
        }
        assert!(
            graph.max_level() >= 1,
            "with 2000 nodes and M=8 there are upper layers"
        );
    }

    #[test]
    fn finds_exact_match_first() {
        let store = clustered(1000, 16, 3);
        let graph = build(&store, HnswParams::default(), |_| {});
        for id in [0u32, 17, 500, 999] {
            let (hits, stats) = search(&graph, &store, store.get(id), 1, 32);
            assert_eq!(hits[0].id, id);
            assert!(
                stats.distance_evals < 1000,
                "visited fewer nodes than brute force"
            );
        }
    }

    #[test]
    fn empty_and_tiny() {
        let store = VecStore::new(4);
        let graph = build(&store, HnswParams::default(), |_| {});
        assert!(
            search(&graph, &store, &[1.0, 0.0, 0.0, 0.0], 5, 10)
                .0
                .is_empty()
        );

        let mut store = VecStore::new(2);
        store.push(&[1.0, 0.0]);
        store.push(&[0.0, 1.0]);
        let graph = build(&store, HnswParams::default(), |_| {});
        let (hits, _) = search(&graph, &store, &[0.1, 1.0], 5, 10);
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), [1, 0]);
    }
}
